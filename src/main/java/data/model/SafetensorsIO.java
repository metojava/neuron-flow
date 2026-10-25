package data.model;

import java.io.BufferedOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.Set;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads and writes the safetensors format: an 8-byte little-endian header
 * length, a JSON header describing each tensor, then the raw tensor bytes.
 *
 * <p>Reading accepts F32, F16, BF16 and F64 tensors and converts them to float.
 * Writing always stores F32. The whole file is loaded into memory.
 */
public final class SafetensorsIO {

    private static final String METADATA_KEY = "__metadata__";

    private SafetensorsIO() {
    }

    public static WeightFile read(Path path) throws IOException {
        byte[] all = Files.readAllBytes(path);
        if (all.length < 8) {
            throw new IOException("Not a safetensors file: too small");
        }
        long headerLen = ByteBuffer.wrap(all, 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();
        if (headerLen < 0 || headerLen > all.length - 8) {
            throw new IOException("Invalid safetensors header length " + headerLen);
        }
        String json = new String(all, 8, (int) headerLen, StandardCharsets.UTF_8);
        Map<String, Object> header = MiniJson.asMap(MiniJson.parse(json));
        long dataStart = 8 + headerLen;

        Map<String, String> metadata = new LinkedHashMap<>();
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : header.entrySet()) {
            String name = e.getKey();
            if (METADATA_KEY.equals(name)) {
                for (Map.Entry<String, Object> m : MiniJson.asMap(e.getValue()).entrySet()) {
                    metadata.put(m.getKey(), String.valueOf(m.getValue()));
                }
                continue;
            }
            Map<String, Object> info = MiniJson.asMap(e.getValue());
            String dtype = String.valueOf(info.get("dtype"));
            int[] shape = toInts(info.get("shape"));
            List<?> offsets = (List<?>) info.get("data_offsets");
            if (offsets == null || offsets.size() != 2) {
                throw new IOException("Tensor " + name + ": bad data_offsets");
            }
            long begin = ((Number) offsets.get(0)).longValue();
            long end = ((Number) offsets.get(1)).longValue();
            if (begin < 0 || end < begin || dataStart + end > all.length) {
                throw new IOException("Tensor " + name + ": data_offsets out of range");
            }
            float[] data = DTypes.decode(dtype, all, (int) (dataStart + begin), (int) (end - begin));
            tensors.put(name, new Tensor(name, shape, data));
        }
        return new WeightFile(metadata, tensors);
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
     * Streaming writer. The header is written when the writer is created, from the
     * shapes alone, so tensors can be written one at a time in the declared order.
     * Each tensor's values must match its declared shape. Closing a writer before all
     * tensors are written deletes the partial file and throws.
     */
    public static final class StreamWriter implements Closeable {

        private final Path path;
        private final List<String> names = new ArrayList<>();
        private final List<Integer> counts = new ArrayList<>();
        private final OutputStream out;
        private int next;
        private boolean closed;

        /**
         * @param path     output file (replaced if it exists)
         * @param shapes   tensor name to shape, in the order they will be written
         * @param metadata string metadata stored in the header (may be empty)
         * @throws IllegalArgumentException on negative dimensions
         */
        public StreamWriter(Path path, Map<String, int[]> shapes, Map<String, String> metadata)
                throws IOException {
            this.path = path;
            StringBuilder header = new StringBuilder("{");
            if (metadata != null && !metadata.isEmpty()) {
                header.append(MiniJson.quote(METADATA_KEY)).append(":{");
                boolean first = true;
                for (Map.Entry<String, String> m : metadata.entrySet()) {
                    if (!first) header.append(',');
                    first = false;
                    header.append(MiniJson.quote(m.getKey())).append(':').append(MiniJson.quote(m.getValue()));
                }
                header.append("},");
            }
            long offset = 0;
            for (Map.Entry<String, int[]> e : shapes.entrySet()) {
                long n = 1;
                for (int d : e.getValue()) {
                    if (d < 0) throw new IllegalArgumentException("Tensor " + e.getKey() + ": negative dimension");
                    n *= d;
                }
                long bytes = 4L * n;
                names.add(e.getKey());
                counts.add((int) n);
                header.append(MiniJson.quote(e.getKey())).append(":{\"dtype\":\"F32\",\"shape\":[");
                int[] shape = e.getValue();
                for (int i = 0; i < shape.length; i++) {
                    if (i > 0) header.append(',');
                    header.append(shape[i]);
                }
                header.append("],\"data_offsets\":[").append(offset).append(',')
                        .append(offset + bytes).append("]},");
                offset += bytes;
            }
            if (header.charAt(header.length() - 1) == ',') {
                header.setLength(header.length() - 1);
            }
            header.append('}');

            // The spec allows padding the header with spaces to an 8-byte boundary.
            byte[] headerBytes = header.toString().getBytes(StandardCharsets.UTF_8);
            int padded = (headerBytes.length + 7) / 8 * 8;
            byte[] headerPadded = new byte[padded];
            java.util.Arrays.fill(headerPadded, (byte) ' ');
            System.arraycopy(headerBytes, 0, headerPadded, 0, headerBytes.length);

            this.out = new BufferedOutputStream(Files.newOutputStream(path), 1 << 16);
            try {
                out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(padded).array());
                out.write(headerPadded);
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
            out.write(DTypes.encodeF32(data));
            next++;
        }

        @Override
        public void close() throws IOException {
            if (closed) return;
            closed = true;
            out.close();
            if (next != names.size()) {
                Files.deleteIfExists(path);
                throw new IOException("Incomplete safetensors file: wrote " + next + " of "
                        + names.size() + " tensors");
            }
        }
    }

    /**
     * Lazy reader: parses only the header when opened and reads each tensor's
     * bytes from the file when {@link #load} is called. Use this for files too
     * large to hold in memory at once.
     */
    public static final class Reader implements Closeable {

        private record Entry(String dtype, int[] shape, long begin, long end) {
        }

        private final FileChannel channel;
        private final long dataStart;
        private final Map<String, Entry> entries = new LinkedHashMap<>();
        private final Map<String, String> metadata = new LinkedHashMap<>();

        public Reader(Path path) throws IOException {
            channel = FileChannel.open(path, StandardOpenOption.READ);
            try {
                long size = channel.size();
                if (size < 8) throw new IOException("Not a safetensors file: too small");
                ByteBuffer lenBuf = ByteBuffer.allocate(8);
                readFully(lenBuf, 0);
                long headerLen = lenBuf.order(ByteOrder.LITTLE_ENDIAN).getLong(0);
                if (headerLen < 0 || headerLen > size - 8) {
                    throw new IOException("Invalid safetensors header length " + headerLen);
                }
                byte[] hb = new byte[(int) headerLen];
                readFully(ByteBuffer.wrap(hb), 8);
                dataStart = 8 + headerLen;
                Map<String, Object> header = MiniJson.asMap(
                        MiniJson.parse(new String(hb, StandardCharsets.UTF_8)));
                for (Map.Entry<String, Object> e : header.entrySet()) {
                    if (METADATA_KEY.equals(e.getKey())) {
                        for (Map.Entry<String, Object> m : MiniJson.asMap(e.getValue()).entrySet()) {
                            metadata.put(m.getKey(), String.valueOf(m.getValue()));
                        }
                        continue;
                    }
                    Map<String, Object> info = MiniJson.asMap(e.getValue());
                    List<?> offsets = (List<?>) info.get("data_offsets");
                    if (offsets == null || offsets.size() != 2) {
                        throw new IOException("Tensor " + e.getKey() + ": bad data_offsets");
                    }
                    long begin = ((Number) offsets.get(0)).longValue();
                    long end = ((Number) offsets.get(1)).longValue();
                    if (begin < 0 || end < begin || dataStart + end > size) {
                        throw new IOException("Tensor " + e.getKey() + ": data_offsets out of range");
                    }
                    entries.put(e.getKey(), new Entry(String.valueOf(info.get("dtype")),
                            toInts(info.get("shape")), begin, end));
                }
            } catch (IOException | RuntimeException e) {
                channel.close();
                throw e;
            }
        }

        /** @return tensor names in file order (metadata excluded) */
        public Set<String> names() {
            return Collections.unmodifiableSet(entries.keySet());
        }

        /** @return string metadata from the header */
        public Map<String, String> metadata() {
            return Collections.unmodifiableMap(metadata);
        }

        /**
         * Reads and decodes one tensor.
         *
         * @param name tensor name
         * @return the tensor as float32
         * @throws IllegalArgumentException if the name is not in the file
         */
        public Tensor load(String name) throws IOException {
            Entry e = entries.get(name);
            if (e == null) throw new IllegalArgumentException("No tensor named " + name);
            int len = (int) (e.end() - e.begin());
            byte[] raw = new byte[len];
            readFully(ByteBuffer.wrap(raw), dataStart + e.begin());
            return new Tensor(name, e.shape(), DTypes.decode(e.dtype(), raw, 0, len));
        }

        private void readFully(ByteBuffer bb, long position) throws IOException {
            long pos = position;
            while (bb.hasRemaining()) {
                int r = channel.read(bb, pos);
                if (r < 0) throw new IOException("Unexpected end of safetensors file");
                pos += r;
            }
        }

        @Override
        public void close() throws IOException {
            channel.close();
        }
    }

    private static int[] toInts(Object o) throws IOException {
        if (!(o instanceof List)) throw new IOException("Expected shape array");
        List<?> list = (List<?>) o;
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) list.get(i)).intValue();
        }
        return out;
    }
}
