package data.model;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.channels.FileChannel;
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
 * Reads and writes GGUF (versions 2 and 3) weight files.
 *
 * <p>Reading supports F32, F16, BF16 and the legacy quantized types Q4_0, Q4_1,
 * Q5_0, Q5_1, Q8_0 and Q8_1 (dequantized to float32). K-quants are rejected.
 * Writing stores F32 tensors and string metadata.
 *
 * <p>GGUF lists dimensions innermost first, so they are reversed here to give
 * row-major shapes.
 */
public final class GgufIO {

    private static final int MAGIC = 0x46554747; // "GGUF" read as little-endian
    private static final int ALIGNMENT_DEFAULT = 32;
    private static final String ALIGNMENT_KEY = "general.alignment";

    // GGUF metadata value types
    private static final int T_UINT8 = 0, T_INT8 = 1, T_UINT16 = 2, T_INT16 = 3,
            T_UINT32 = 4, T_INT32 = 5, T_FLOAT32 = 6, T_BOOL = 7, T_STRING = 8,
            T_ARRAY = 9, T_UINT64 = 10, T_INT64 = 11, T_FLOAT64 = 12;

    private GgufIO() {
    }

    private record Info(String name, int[] shape, int type, long offset, long numel) {
    }

    /**
     * Lazy reader: parses the header when opened; each tensor's bytes are read
     * from the file only when {@link #load} is called.
     */
    public static final class Reader implements Closeable {

        private final FileChannel channel;
        private final ChannelCursor cursor;
        private final long dataStart;
        private final Map<String, String> metadata = new LinkedHashMap<>();
        private final Map<String, Info> infos = new LinkedHashMap<>();

        public Reader(Path path) throws IOException {
            FileChannel ch = FileChannel.open(path, StandardOpenOption.READ);
            try {
                this.channel = ch;
                this.cursor = new ChannelCursor(ch);
                if (cursor.size() < 24) throw new IOException("Not a GGUF file");
                ByteBuffer h = cursor.le(24);
                if (h.getInt() != MAGIC) throw new IOException("Not a GGUF file");
                int version = h.getInt();
                if (version < 2 || version > 3) {
                    throw new IOException("Unsupported GGUF version " + version);
                }
                long nTensors = h.getLong();
                long nKv = h.getLong();

                Map<String, Object> kv = new LinkedHashMap<>();
                for (long i = 0; i < nKv; i++) {
                    String key = readString(cursor);
                    int type = cursor.le(4).getInt();
                    kv.put(key, readValue(cursor, type));
                }
                int alignment = ALIGNMENT_DEFAULT;
                if (kv.get(ALIGNMENT_KEY) instanceof Number n) {
                    alignment = n.intValue();
                }
                for (Map.Entry<String, Object> e : kv.entrySet()) {
                    metadata.put(e.getKey(), String.valueOf(e.getValue()));
                }

                for (long i = 0; i < nTensors; i++) {
                    String name = readString(cursor);
                    int nDims = cursor.le(4).getInt();
                    if (nDims < 0 || nDims > 8) throw new IOException("Tensor " + name + ": bad rank");
                    long[] dims = new long[nDims];
                    for (int d = 0; d < nDims; d++) dims[d] = cursor.le(8).getLong();
                    ByteBuffer t = cursor.le(12);
                    int type = t.getInt();
                    long offset = t.getLong();
                    if (!GgufQuant.isSupported(type)) throw GgufQuant.unsupported(type);
                    int[] shape = new int[nDims];
                    long numel = 1;
                    for (int d = 0; d < nDims; d++) {
                        shape[d] = (int) dims[nDims - 1 - d];
                        numel *= dims[d];
                    }
                    infos.put(name, new Info(name, shape, type, offset, numel));
                }
                this.dataStart = align(cursor.position(), alignment);
                for (Info info : infos.values()) {
                    long bytes = GgufQuant.byteCount(info.type(), info.numel());
                    if (dataStart + info.offset() + bytes > cursor.size()) {
                        throw new IOException("Tensor " + info.name() + " extends past end of file");
                    }
                }
            } catch (IOException | RuntimeException e) {
                ch.close();
                throw e;
            }
        }

        /** @return tensor names in file order */
        public Set<String> names() {
            return Collections.unmodifiableSet(infos.keySet());
        }

        /** @return string metadata from the header */
        public Map<String, String> metadata() {
            return Collections.unmodifiableMap(metadata);
        }

        /**
         * Reads and dequantizes one tensor.
         *
         * @param name tensor name
         * @return the tensor as float32, shaped row-major
         * @throws IllegalArgumentException if the name is not in the file
         */
        public Tensor load(String name) throws IOException {
            Info info = infos.get(name);
            if (info == null) throw new IllegalArgumentException("No tensor named " + name);
            long bytes = GgufQuant.byteCount(info.type(), info.numel());
            byte[] raw = cursor.readAt(dataStart + info.offset(), bytes);
            return new Tensor(name, info.shape(),
                    GgufQuant.dequantize(info.type(), raw, 0, (int) info.numel()));
        }

        @Override
        public void close() throws IOException {
            channel.close();
        }
    }

    /**
     * Reads every tensor into memory. Use {@link Reader} for lazy access.
     *
     * @param path GGUF file
     * @return metadata and all tensors
     * @throws IOException if the file is malformed or uses an unsupported type
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
     * Writes a weight file. Tensors are streamed one at a time through
     * {@link StreamWriter}, so the output is not built in memory.
     *
     * @param path output file (replaced if it exists)
     * @param file metadata and tensors to write (stored as F32)
     */
    public static void write(Path path, WeightFile file) throws IOException {
        Map<String, int[]> shapes = new LinkedHashMap<>();
        for (Tensor t : file.tensors().values()) {
            shapes.put(t.name(), t.shape());
        }
        try (StreamWriter w = new StreamWriter(path, shapes, file.metadata())) {
            for (Tensor t : file.tensors().values()) {
                w.write(t.name(), t.data());
            }
        }
    }

    /**
     * Streaming GGUF writer (version 3, F32 tensors). The header and tensor offsets are
     * computed from the shapes when the writer is created. Tensors are then written in
     * the declared order, each padded to its 32-byte aligned offset. Closing a writer
     * before all tensors are written deletes the partial file and throws.
     */
    public static final class StreamWriter implements Closeable {

        private final Path path;
        private final List<String> names = new ArrayList<>();
        private final List<Integer> counts = new ArrayList<>();
        private final List<Long> offsets = new ArrayList<>();
        private final OutputStream out;
        private long pos;   // bytes written since the start of the data section
        private int next;
        private boolean closed;

        /**
         * @param path     output file (replaced if it exists)
         * @param shapes   tensor name to shape, in the order they will be written
         * @param metadata string metadata (the alignment key is reserved)
         * @throws IllegalArgumentException on negative dimensions
         */
        public StreamWriter(Path path, Map<String, int[]> shapes, Map<String, String> metadata)
                throws IOException {
            this.path = path;
            ByteArrayOutputStream hdr = new ByteArrayOutputStream();
            u32(hdr, MAGIC);
            u32(hdr, 3);
            u64(hdr, shapes.size());

            int userKeys = 0;
            if (metadata != null) {
                for (String k : metadata.keySet()) {
                    if (!ALIGNMENT_KEY.equals(k)) userKeys++;
                }
            }
            u64(hdr, userKeys + 1);
            str(hdr, ALIGNMENT_KEY);
            u32(hdr, T_UINT32);
            u32(hdr, ALIGNMENT_DEFAULT);
            if (metadata != null) {
                for (Map.Entry<String, String> m : metadata.entrySet()) {
                    if (ALIGNMENT_KEY.equals(m.getKey())) continue;
                    str(hdr, m.getKey());
                    u32(hdr, T_STRING);
                    str(hdr, m.getValue());
                }
            }

            long running = 0;
            for (Map.Entry<String, int[]> e : shapes.entrySet()) {
                long n = 1;
                for (int d : e.getValue()) {
                    if (d < 0) throw new IllegalArgumentException("Tensor " + e.getKey() + ": negative dimension");
                    n *= d;
                }
                running = align(running, ALIGNMENT_DEFAULT);
                names.add(e.getKey());
                counts.add((int) n);
                offsets.add(running);
                running += 4L * n;
            }
            int idx = 0;
            for (Map.Entry<String, int[]> e : shapes.entrySet()) {
                int[] shape = e.getValue();
                str(hdr, e.getKey());
                u32(hdr, shape.length);
                for (int d = shape.length - 1; d >= 0; d--) u64(hdr, shape[d]);
                u32(hdr, GgufQuant.F32);
                u64(hdr, offsets.get(idx++));
            }

            this.out = new BufferedOutputStream(Files.newOutputStream(path), 1 << 16);
            try {
                byte[] headerBytes = hdr.toByteArray();
                out.write(headerBytes);
                long padding = align(headerBytes.length, ALIGNMENT_DEFAULT) - headerBytes.length;
                out.write(new byte[(int) padding]);
            } catch (IOException e) {
                out.close();
                throw e;
            }
        }

        /** @return number of tensors written so far */
        public int tensorsWritten() {
            return next;
        }

        /**
         * Writes the next tensor.
         *
         * @param name tensor name; must match the next name in the declared order
         * @param data values in row-major order; length must match the declared shape
         * @throws IllegalArgumentException on an out-of-order name or wrong length
         */
        public void write(String name, float[] data) throws IOException {
            if (closed) throw new IllegalStateException("Writer is closed");
            if (next >= names.size()) {
                throw new IllegalArgumentException("All " + names.size() + " tensors already written");
            }
            if (!names.get(next).equals(name)) {
                throw new IllegalArgumentException("Expected tensor " + names.get(next) + " but got " + name);
            }
            if (data.length != counts.get(next)) {
                throw new IllegalArgumentException("Tensor " + name + ": expected " + counts.get(next)
                        + " values, got " + data.length);
            }
            long target = offsets.get(next);
            if (target > pos) {
                out.write(new byte[(int) (target - pos)]);
                pos = target;
            }
            byte[] raw = DTypes.encodeF32(data);
            out.write(raw);
            pos += raw.length;
            next++;
        }

        @Override
        public void close() throws IOException {
            if (closed) return;
            closed = true;
            out.close();
            if (next != names.size()) {
                Files.deleteIfExists(path);
                throw new IOException("Incomplete GGUF file: wrote " + next + " of "
                        + names.size() + " tensors");
            }
        }
    }

    // ---- reading helpers ----

    private static String readString(ChannelCursor c) throws IOException {
        long len = c.le(8).getLong();
        if (len < 0 || len > c.size()) throw new IOException("Bad string length");
        return new String(c.read((int) len), StandardCharsets.UTF_8);
    }

    private static Object readValue(ChannelCursor c, int type) throws IOException {
        switch (type) {
            case T_UINT8: return c.le(1).get() & 0xff;
            case T_INT8: return (int) c.le(1).get();
            case T_UINT16: return c.le(2).getShort() & 0xffff;
            case T_INT16: return (int) c.le(2).getShort();
            case T_UINT32: return c.le(4).getInt() & 0xffffffffL;
            case T_INT32: return c.le(4).getInt();
            case T_FLOAT32: return c.le(4).getFloat();
            case T_BOOL: return c.le(1).get() != 0;
            case T_STRING: return readString(c);
            case T_UINT64: return c.le(8).getLong();
            case T_INT64: return c.le(8).getLong();
            case T_FLOAT64: return c.le(8).getDouble();
            case T_ARRAY: {
                ByteBuffer hdr = c.le(12);
                int elemType = hdr.getInt();
                long count = hdr.getLong();
                if (count < 0 || count > c.size()) throw new IOException("Bad array length");
                List<Object> list = new ArrayList<>();
                for (long i = 0; i < count; i++) list.add(readValue(c, elemType));
                return list;
            }
            default:
                throw new IOException("Unknown GGUF value type " + type);
        }
    }

    // ---- writing helpers ----

    private static void u32(ByteArrayOutputStream o, long v) {
        for (int i = 0; i < 4; i++) o.write((int) (v >>> (8 * i)) & 0xff);
    }

    private static void u64(ByteArrayOutputStream o, long v) {
        for (int i = 0; i < 8; i++) o.write((int) (v >>> (8 * i)) & 0xff);
    }

    private static void str(ByteArrayOutputStream o, String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        u64(o, b.length);
        o.write(b, 0, b.length);
    }

    private static void padTo(ByteArrayOutputStream o, int alignment) {
        while (o.size() % alignment != 0) o.write(0);
    }

    private static long align(long x, int alignment) {
        return (x + alignment - 1) / alignment * alignment;
    }
}
