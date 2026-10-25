package data.model;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Reads and writes the weights (graph initializers) of an ONNX model.
 *
 * <p>This is a minimal protobuf reader and writer. It does not read or write the
 * computation graph, so the output of {@link #write} is a weights container,
 * not a runnable ONNX model. Reading accepts FLOAT, FLOAT16, BFLOAT16 and DOUBLE
 * initializers (from raw_data, or FLOAT from float_data); other types are rejected.
 *
 * <p>Protobuf field numbers used (from onnx.proto):
 * ModelProto.graph = 7; GraphProto.name = 2, initializer = 5;
 * TensorProto.dims = 1, data_type = 2, float_data = 4, name = 8, raw_data = 9.
 */
public final class OnnxIO {

    private static final int MODEL_IR_VERSION = 1;
    private static final int MODEL_GRAPH = 7;
    private static final int GRAPH_NAME = 2;
    private static final int GRAPH_INITIALIZER = 5;
    private static final int GRAPH_NODE = 1;
    private static final int GRAPH_INPUT = 11;
    private static final int GRAPH_OUTPUT = 12;
    private static final int MODEL_OPSET_IMPORT = 8;
    private static final int DEFAULT_OPSET = 17;
    private static final int TENSOR_DIMS = 1;
    private static final int TENSOR_DATA_TYPE = 2;
    private static final int TENSOR_FLOAT_DATA = 4;
    private static final int TENSOR_NAME = 8;
    private static final int TENSOR_RAW_DATA = 9;

    private static final int DT_FLOAT = 1, DT_DOUBLE = 11, DT_FLOAT16 = 10, DT_BFLOAT16 = 16;

    private OnnxIO() {
    }

    /**
     * Where one initializer's data lives. Either {@code inline} holds the values
     * (unpacked float_data), or the bytes at {@code offset}/{@code length} are decoded
     * as {@code dtype}.
     */
    private record Entry(int[] shape, String dtype, long offset, long length, float[] inline) {
    }

    /**
     * Lazy reader: scans the protobuf structure when opened and records where each
     * initializer's data is stored. Tensor bytes are read from the file only when
     * {@link #load} is called.
     */
    public static final class Reader implements Closeable {

        private final FileChannel channel;
        private final ChannelCursor cur;
        private final Map<String, String> metadata = new LinkedHashMap<>();
        private final Map<String, Entry> entries = new LinkedHashMap<>();
        private final List<OnnxGraph.Node> nodes = new ArrayList<>();
        private final List<OnnxGraph.ValueInfo> inputs = new ArrayList<>();
        private final List<OnnxGraph.ValueInfo> outputs = new ArrayList<>();

        /** Element type and shape of a TypeProto. */
        private record TypeInfo(int elemType, long[] shape) {
        }

        public Reader(Path path) throws IOException {
            FileChannel ch = FileChannel.open(path, StandardOpenOption.READ);
            try {
                this.channel = ch;
                this.cur = new ChannelCursor(ch);
                scanModel(cur.size());
            } catch (IOException | RuntimeException e) {
                ch.close();
                throw e;
            }
        }

        /** @return initializer names in file order */
        public Set<String> names() {
            return Collections.unmodifiableSet(entries.keySet());
        }

        /** @return metadata such as graph_name */
        public Map<String, String> metadata() {
            return Collections.unmodifiableMap(metadata);
        }

        /**
         * Reads and decodes one initializer.
         *
         * @param name initializer name
         * @return the tensor as float32
         * @throws IllegalArgumentException if the name is not in the file
         */
        public Tensor load(String name) throws IOException {
            Entry e = entries.get(name);
            if (e == null) throw new IllegalArgumentException("No initializer named " + name);
            if (e.inline() != null) {
                return new Tensor(name, e.shape(), e.inline().clone());
            }
            byte[] raw = cur.readAt(e.offset(), e.length());
            return new Tensor(name, e.shape(), DTypes.decode(e.dtype(), raw, 0, raw.length));
        }

        private void scanModel(long end) throws IOException {
            cur.seek(0);
            while (cur.position() < end) {
                long key = varint();
                int field = (int) (key >>> 3);
                int wire = (int) (key & 7);
                if (field == MODEL_GRAPH && wire == 2) {
                    long len = varint();
                    scanGraph(checkedEnd(len, end));
                } else {
                    skip(wire, end);
                }
            }
        }

        private void scanGraph(long end) throws IOException {
            while (cur.position() < end) {
                long key = varint();
                int field = (int) (key >>> 3);
                int wire = (int) (key & 7);
                if (field == GRAPH_INITIALIZER && wire == 2) {
                    long len = varint();
                    scanTensor(checkedEnd(len, end), entries.size());
                } else if (field == GRAPH_NODE && wire == 2) {
                    long len = varint();
                    nodes.add(scanNode(checkedEnd(len, end)));
                } else if (field == GRAPH_INPUT && wire == 2) {
                    long len = varint();
                    inputs.add(scanValueInfo(checkedEnd(len, end)));
                } else if (field == GRAPH_OUTPUT && wire == 2) {
                    long len = varint();
                    outputs.add(scanValueInfo(checkedEnd(len, end)));
                } else if (field == GRAPH_NAME && wire == 2) {
                    long len = varint();
                    metadata.put("graph_name", new String(
                            cur.readAt(cur.position(), len), StandardCharsets.UTF_8));
                    cur.seek(cur.position() + len);
                } else {
                    skip(wire, end);
                }
            }
        }

        private void scanTensor(long end, int index) throws IOException {
            List<Long> dims = new ArrayList<>();
            int dataType = 0;
            String name = null;
            long rawOff = -1;
            long rawLen = 0;
            long floatOff = -1;
            long floatLen = 0;
            List<Float> unpacked = null;

            while (cur.position() < end) {
                long key = varint();
                int field = (int) (key >>> 3);
                int wire = (int) (key & 7);
                switch (field) {
                    case TENSOR_DIMS:
                        if (wire == 2) {
                            long len = varint();
                            long dEnd = checkedEnd(len, end);
                            while (cur.position() < dEnd) dims.add(varint());
                        } else {
                            dims.add(varint());
                        }
                        break;
                    case TENSOR_DATA_TYPE:
                        dataType = (int) varint();
                        break;
                    case TENSOR_FLOAT_DATA:
                        if (wire == 2) {
                            long len = varint();
                            floatOff = cur.position();
                            floatLen = len;
                            skipBytes(len, end);
                        } else {
                            if (unpacked == null) unpacked = new ArrayList<>();
                            unpacked.add(cur.le(4).getFloat());
                        }
                        break;
                    case TENSOR_NAME:
                        if (wire != 2) throw new IOException("Bad tensor name field");
                        long nLen = varint();
                        name = new String(cur.readAt(cur.position(), nLen), StandardCharsets.UTF_8);
                        skipBytes(nLen, end);
                        break;
                    case TENSOR_RAW_DATA:
                        if (wire != 2) throw new IOException("Bad raw_data field");
                        rawLen = varint();
                        rawOff = cur.position();
                        skipBytes(rawLen, end);
                        break;
                    default:
                        skip(wire, end);
                }
            }

            if (name == null || name.isEmpty()) name = "initializer_" + index;
            int[] shape = new int[dims.size()];
            long numel = 1;
            for (int i = 0; i < shape.length; i++) {
                shape[i] = (int) (long) dims.get(i);
                numel *= shape[i];
            }

            Entry entry;
            if (unpacked != null) {
                if (dataType != DT_FLOAT) throw new IOException("Tensor " + name + ": unpacked data must be FLOAT");
                float[] values = new float[unpacked.size()];
                for (int i = 0; i < values.length; i++) values[i] = unpacked.get(i);
                entry = new Entry(shape, "F32", -1, 0, values);
            } else if (rawOff >= 0) {
                String dtype = onnxDtype(dataType, name);
                if (rawLen != numel * DTypes.elementSize(dtype)) {
                    throw new IOException("Tensor " + name + ": raw_data length does not match shape");
                }
                entry = new Entry(shape, dtype, rawOff, rawLen, null);
            } else if (floatOff >= 0) {
                if (dataType != DT_FLOAT) throw new IOException("Tensor " + name + ": float_data must be FLOAT");
                if (floatLen != numel * 4) {
                    throw new IOException("Tensor " + name + ": float_data length does not match shape");
                }
                entry = new Entry(shape, "F32", floatOff, floatLen, null);
            } else {
                throw new IOException("Tensor " + name + ": no data");
            }
            entries.put(name, entry);
        }

        /** Snapshot of the graph structure read so far. */
        public OnnxGraph graph() {
            return new OnnxGraph(metadata.getOrDefault("graph_name", ""),
                    List.copyOf(inputs), List.copyOf(outputs), List.copyOf(nodes),
                    List.copyOf(entries.keySet()));
        }

        private OnnxGraph.Node scanNode(long end) throws IOException {
            String name = "";
            String opType = "";
            List<String> ins = new ArrayList<>();
            List<String> outs = new ArrayList<>();
            Map<String, Object> attrs = new LinkedHashMap<>();
            while (cur.position() < end) {
                long key = varint();
                int field = (int) (key >>> 3);
                int wire = (int) (key & 7);
                if (field == 1 && wire == 2) {
                    ins.add(readString(end));
                } else if (field == 2 && wire == 2) {
                    outs.add(readString(end));
                } else if (field == 3 && wire == 2) {
                    name = readString(end);
                } else if (field == 4 && wire == 2) {
                    opType = readString(end);
                } else if (field == 5 && wire == 2) {
                    long len = varint();
                    Map.Entry<String, Object> a = scanAttribute(checkedEnd(len, end));
                    attrs.put(a.getKey(), a.getValue());
                } else {
                    skip(wire, end);
                }
            }
            return new OnnxGraph.Node(name, opType, ins, outs, attrs);
        }

        private Map.Entry<String, Object> scanAttribute(long end) throws IOException {
            String name = "";
            int type = 0;
            Object value = null;
            List<Float> floats = new ArrayList<>();
            List<Long> ints = new ArrayList<>();
            List<String> strings = new ArrayList<>();
            boolean hasFloats = false;
            boolean hasInts = false;
            boolean hasStrings = false;
            while (cur.position() < end) {
                long key = varint();
                int field = (int) (key >>> 3);
                int wire = (int) (key & 7);
                switch (field) {
                    case 1:
                        if (wire != 2) throw new IOException("Bad attribute name");
                        name = readString(end);
                        break;
                    case 2:
                        if (wire != 5) throw new IOException("Bad attribute float");
                        value = cur.le(4).getFloat();
                        break;
                    case 3:
                        value = varint();
                        break;
                    case 4:
                        value = readString(end);
                        break;
                    case 7:
                        hasFloats = true;
                        if (wire == 2) {
                            long len = varint();
                            long limit = checkedEnd(len, end);
                            while (cur.position() < limit) floats.add(cur.le(4).getFloat());
                        } else {
                            floats.add(cur.le(4).getFloat());
                        }
                        break;
                    case 8:
                        hasInts = true;
                        if (wire == 2) {
                            long len = varint();
                            long limit = checkedEnd(len, end);
                            while (cur.position() < limit) ints.add(varint());
                        } else {
                            ints.add(varint());
                        }
                        break;
                    case 9:
                        hasStrings = true;
                        strings.add(readString(end));
                        break;
                    case 20:
                        type = (int) varint();
                        break;
                    default:
                        // TENSOR, GRAPH, SPARSE_TENSOR and their lists are not decoded.
                        skip(wire, end);
                }
            }
            if (hasFloats) value = floats;
            if (hasInts) value = ints;
            if (hasStrings) value = strings;
            if (value == null && type == 6) value = floats;
            if (value == null && type == 7) value = ints;
            if (value == null && type == 8) value = strings;
            return new java.util.AbstractMap.SimpleImmutableEntry<>(name, value);
        }

        private OnnxGraph.ValueInfo scanValueInfo(long end) throws IOException {
            String name = "";
            TypeInfo type = new TypeInfo(0, new long[0]);
            while (cur.position() < end) {
                long key = varint();
                int field = (int) (key >>> 3);
                int wire = (int) (key & 7);
                if (field == 1 && wire == 2) {
                    name = readString(end);
                } else if (field == 2 && wire == 2) {
                    long len = varint();
                    type = scanType(checkedEnd(len, end));
                } else {
                    skip(wire, end);
                }
            }
            return new OnnxGraph.ValueInfo(name, type.elemType(), type.shape());
        }

        private TypeInfo scanType(long end) throws IOException {
            TypeInfo result = new TypeInfo(0, new long[0]);
            while (cur.position() < end) {
                long key = varint();
                int field = (int) (key >>> 3);
                int wire = (int) (key & 7);
                if (field == 1 && wire == 2) {
                    long len = varint();
                    result = scanTensorType(checkedEnd(len, end));
                } else {
                    skip(wire, end);
                }
            }
            return result;
        }

        private TypeInfo scanTensorType(long end) throws IOException {
            int elemType = 0;
            long[] shape = new long[0];
            while (cur.position() < end) {
                long key = varint();
                int field = (int) (key >>> 3);
                int wire = (int) (key & 7);
                if (field == 1 && wire == 0) {
                    elemType = (int) varint();
                } else if (field == 2 && wire == 2) {
                    long len = varint();
                    shape = scanShape(checkedEnd(len, end));
                } else {
                    skip(wire, end);
                }
            }
            return new TypeInfo(elemType, shape);
        }

        private long[] scanShape(long end) throws IOException {
            List<Long> dims = new ArrayList<>();
            while (cur.position() < end) {
                long key = varint();
                int field = (int) (key >>> 3);
                int wire = (int) (key & 7);
                if (field == 1 && wire == 2) {
                    long len = varint();
                    dims.add(scanDimension(checkedEnd(len, end)));
                } else {
                    skip(wire, end);
                }
            }
            long[] out = new long[dims.size()];
            for (int i = 0; i < out.length; i++) out[i] = dims.get(i);
            return out;
        }

        /** dim_value if present; a symbolic dim_param or no value gives -1. */
        private long scanDimension(long end) throws IOException {
            long value = -1;
            while (cur.position() < end) {
                long key = varint();
                int field = (int) (key >>> 3);
                int wire = (int) (key & 7);
                if (field == 1 && wire == 0) {
                    value = varint();
                } else if (field == 2 && wire == 2) {
                    readString(end);
                    value = -1;
                } else {
                    skip(wire, end);
                }
            }
            return value;
        }

        private String readString(long limit) throws IOException {
            long len = varint();
            checkedEnd(len, limit);
            String s = new String(cur.readAt(cur.position(), len), StandardCharsets.UTF_8);
            cur.seek(cur.position() + len);
            return s;
        }

        private long varint() throws IOException {
            long result = 0;
            int shift = 0;
            while (true) {
                int b = cur.read(1)[0] & 0xff;
                result |= (long) (b & 0x7f) << shift;
                if ((b & 0x80) == 0) return result;
                shift += 7;
                if (shift > 63) throw new IOException("Protobuf varint too long");
            }
        }

        private long checkedEnd(long len, long limit) throws IOException {
            if (len < 0 || cur.position() + len > limit) {
                throw new IOException("Bad protobuf length");
            }
            return cur.position() + len;
        }

        private void skipBytes(long len, long limit) throws IOException {
            checkedEnd(len, limit);
            cur.seek(cur.position() + len);
        }

        private void skip(int wire, long limit) throws IOException {
            switch (wire) {
                case 0: varint(); break;
                case 1: skipBytes(8, limit); break;
                case 2: skipBytes(varint(), limit); break;
                case 5: skipBytes(4, limit); break;
                default: throw new IOException("Unsupported protobuf wire type " + wire);
            }
        }

        @Override
        public void close() throws IOException {
            channel.close();
        }
    }

    /**
     * Reads every initializer into memory. Use {@link Reader} for lazy access.
     *
     * @param path ONNX file
     * @return graph metadata and all initializers
     * @throws IOException on malformed input or unsupported data types
     */
    public static WeightFile read(Path path) throws IOException {
        try (Reader r = new Reader(path)) {
            Map<String, Tensor> tensors = new LinkedHashMap<>();
            for (String name : r.names()) {
                tensors.put(name, r.load(name));
            }
            return new WeightFile(new LinkedHashMap<>(r.metadata()), tensors);
        }
    }

    /**
     * Writes a weights-only model: the tensors as initializers, with no nodes.
     *
     * @param path output file (replaced if it exists)
     * @param file tensors to store (as FLOAT) and the graph name in metadata
     */
    public static void write(Path path, WeightFile file) throws IOException {
        String name = file.metadata().getOrDefault("graph_name", "neuronflow_weights");
        writeGraph(path, new OnnxGraph(name, List.of(), List.of(), List.of(), List.of()), file);
    }

    /**
     * Writes a model with graph structure and weights: nodes (with attributes), graph
     * inputs and outputs, and the tensors of {@code weights} as initializers.
     *
     * <p>Attribute values are encoded from their Java types as described in
     * {@link OnnxGraph.Node}. An empty list is encoded as an INTS attribute, so it reads
     * back as an empty list of Long.
     *
     * @param path    output file (replaced if it exists)
     * @param graph   graph structure; its initializerNames are ignored
     * @param weights initializer values
     * @throws IllegalArgumentException for an attribute value of an unsupported type
     */
    public static void writeGraph(Path path, OnnxGraph graph, WeightFile weights) throws IOException {
        Proto.Writer g = new Proto.Writer();
        g.string(GRAPH_NAME, graph.name());
        for (OnnxGraph.Node node : graph.nodes()) {
            g.bytes(GRAPH_NODE, encodeNode(node).toByteArray());
        }
        for (Tensor t : weights.tensors().values()) {
            Proto.Writer tp = new Proto.Writer();
            for (int d : t.shape()) tp.varint(TENSOR_DIMS, d);
            tp.varint(TENSOR_DATA_TYPE, DT_FLOAT);
            tp.string(TENSOR_NAME, t.name());
            tp.bytes(TENSOR_RAW_DATA, DTypes.encodeF32(t.data()));
            g.bytes(GRAPH_INITIALIZER, tp.toByteArray());
        }
        for (OnnxGraph.ValueInfo vi : graph.inputs()) {
            g.bytes(GRAPH_INPUT, encodeValueInfo(vi).toByteArray());
        }
        for (OnnxGraph.ValueInfo vi : graph.outputs()) {
            g.bytes(GRAPH_OUTPUT, encodeValueInfo(vi).toByteArray());
        }

        Proto.Writer opset = new Proto.Writer();
        opset.varint(2, DEFAULT_OPSET);
        Proto.Writer model = new Proto.Writer();
        model.varint(MODEL_IR_VERSION, 8);
        model.bytes(MODEL_OPSET_IMPORT, opset.toByteArray());
        model.bytes(MODEL_GRAPH, g.toByteArray());
        Files.write(path, model.toByteArray());
    }

    private static Proto.Writer encodeNode(OnnxGraph.Node node) {
        Proto.Writer n = new Proto.Writer();
        for (String s : node.inputs()) n.string(1, s);
        for (String s : node.outputs()) n.string(2, s);
        if (!node.name().isEmpty()) n.string(3, node.name());
        n.string(4, node.opType());
        for (Map.Entry<String, Object> a : node.attributes().entrySet()) {
            n.bytes(5, encodeAttribute(a.getKey(), a.getValue()).toByteArray());
        }
        return n;
    }

    private static Proto.Writer encodeAttribute(String name, Object value) {
        Proto.Writer a = new Proto.Writer();
        a.string(1, name);
        if (value instanceof Float f) {
            a.varint(20, 1);
            a.fixed32(2, f);
        } else if (value instanceof Long l) {
            a.varint(20, 2);
            a.varint(3, l);
        } else if (value instanceof String s) {
            a.varint(20, 3);
            a.string(4, s);
        } else if (value instanceof List<?> list) {
            Object first = list.isEmpty() ? null : list.get(0);
            if (first instanceof Float) {
                a.varint(20, 6);
                for (Object o : list) a.fixed32(7, (Float) o);
            } else if (first instanceof String) {
                a.varint(20, 8);
                for (Object o : list) a.string(9, (String) o);
            } else {
                a.varint(20, 7);
                for (Object o : list) a.varint(8, (Long) o);
            }
        } else {
            throw new IllegalArgumentException("Attribute " + name + ": unsupported value " + value);
        }
        return a;
    }

    private static Proto.Writer encodeValueInfo(OnnxGraph.ValueInfo vi) {
        Proto.Writer shape = new Proto.Writer();
        for (long d : vi.shape()) {
            Proto.Writer dim = new Proto.Writer();
            if (d >= 0) {
                dim.varint(1, d);
            } else {
                dim.string(2, "?");
            }
            shape.bytes(1, dim.toByteArray());
        }
        Proto.Writer tensor = new Proto.Writer();
        tensor.varint(1, vi.elemType());
        tensor.bytes(2, shape.toByteArray());
        Proto.Writer type = new Proto.Writer();
        type.bytes(1, tensor.toByteArray());
        Proto.Writer v = new Proto.Writer();
        v.string(1, vi.name());
        v.bytes(2, type.toByteArray());
        return v;
    }

    private static String onnxDtype(int type, String name) throws IOException {
        switch (type) {
            case DT_FLOAT: return "F32";
            case DT_FLOAT16: return "F16";
            case DT_BFLOAT16: return "BF16";
            case DT_DOUBLE: return "F64";
            default:
                throw new IOException("Tensor " + name + ": unsupported ONNX data type " + type);
        }
    }

    /** Protobuf wire-format writer (used by {@link #write} and by tests). */
    static final class Proto {

        private Proto() {
        }

        static final class Writer {
            private final ByteArrayOutputStream out = new ByteArrayOutputStream();

            void varint(int field, long value) {
                rawVarint((field << 3) | 0L);
                rawVarint(value);
            }

            void fixed32(int field, float value) {
                rawVarint((field << 3) | 5L);
                int bits = Float.floatToRawIntBits(value);
                for (int i = 0; i < 4; i++) out.write((bits >>> (8 * i)) & 0xff);
            }

            void string(int field, String s) {
                bytes(field, s.getBytes(StandardCharsets.UTF_8));
            }

            void bytes(int field, byte[] b) {
                rawVarint((field << 3) | 2L);
                rawVarint(b.length);
                out.write(b, 0, b.length);
            }

            byte[] toByteArray() {
                return out.toByteArray();
            }

            private void rawVarint(long v) {
                while ((v & ~0x7fL) != 0) {
                    out.write((int) ((v & 0x7f) | 0x80));
                    v >>>= 7;
                }
                out.write((int) v);
            }
        }
    }
}
