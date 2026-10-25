package data.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelFormatsTest {

    @TempDir
    Path tmp;

    private static WeightFile sample() {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        tensors.put("dense.weight", new Tensor("dense.weight", new int[] {2, 3},
                new float[] {1, 2, 3, 4, 5, 6}));
        tensors.put("dense.bias", new Tensor("dense.bias", new int[] {3},
                new float[] {0.5f, -0.5f, 0.25f}));
        Map<String, String> meta = new LinkedHashMap<>();
        meta.put("format", "neuronflow");
        return new WeightFile(meta, tensors);
    }

    private static void assertSame(WeightFile expected, WeightFile actual) {
        assertEquals(expected.tensors().keySet(), actual.tensors().keySet());
        for (Tensor t : expected.tensors().values()) {
            Tensor got = actual.tensors().get(t.name());
            assertArrayEquals(t.shape(), got.shape());
            assertArrayEquals(t.data(), got.data(), 0f);
        }
    }

    @Test
    void safetensorsRoundTrip() throws IOException {
        Path file = tmp.resolve("model.safetensors");
        SafetensorsIO.write(file, sample());
        WeightFile back = SafetensorsIO.read(file);
        assertSame(sample(), back);
        assertEquals("neuronflow", back.metadata().get("format"));
    }

    @Test
    void safetensorsReadsF16() throws IOException {
        String header = "{\"w\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[0,4]}}";
        byte[] hb = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer bb = ByteBuffer.allocate(8 + hb.length + 4).order(ByteOrder.LITTLE_ENDIAN);
        bb.putLong(hb.length).put(hb);
        bb.putShort((short) 0x3C00); // 1.0
        bb.putShort((short) 0xC000); // -2.0
        Path file = tmp.resolve("f16.safetensors");
        Files.write(file, bb.array());
        float[] got = SafetensorsIO.read(file).tensors().get("w").data();
        assertArrayEquals(new float[] {1f, -2f}, got, 0f);
    }

    @Test
    void safetensorsRejectsTruncatedData() throws IOException {
        String header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}";
        byte[] hb = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer bb = ByteBuffer.allocate(8 + hb.length + 4).order(ByteOrder.LITTLE_ENDIAN);
        bb.putLong(hb.length).put(hb).putFloat(1f);
        Path file = tmp.resolve("short.safetensors");
        Files.write(file, bb.array());
        assertThrows(IOException.class, () -> SafetensorsIO.read(file));
    }

    @Test
    void ggufRoundTrip() throws IOException {
        Path file = tmp.resolve("model.gguf");
        GgufIO.write(file, sample());
        WeightFile back = GgufIO.read(file);
        assertSame(sample(), back);
        assertEquals("neuronflow", back.metadata().get("format"));
    }

    @Test
    void ggufRejectsQuantizedTensors() throws IOException {
        // Minimal GGUF v3 file with one tensor of GGML type 16 (IQ2_XXS, unsupported).
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        writeLe(o, 0x46554747, 4);
        writeLe(o, 3, 4);
        writeLe(o, 1, 8);  // tensor count
        writeLe(o, 0, 8);  // kv count
        writeStr(o, "t");
        writeLe(o, 1, 4);  // n_dims
        writeLe(o, 32, 8); // dim
        writeLe(o, 16, 4); // type IQ2_XXS
        writeLe(o, 0, 8);  // offset
        Path file = tmp.resolve("q4.gguf");
        Files.write(file, o.toByteArray());
        IOException e = assertThrows(IOException.class, () -> GgufIO.read(file));
        assertTrue(e.getMessage().contains("Unsupported GGML tensor type"));
    }

    @Test
    void onnxRoundTrip() throws IOException {
        Path file = tmp.resolve("weights.onnx");
        OnnxIO.write(file, sample());
        WeightFile back = OnnxIO.read(file);
        assertSame(sample(), back);
    }

    @Test
    void onnxReadsRawFloat16Initializer() throws IOException {
        // TensorProto: dims=[2], data_type=10 (FLOAT16), name="h", raw_data=[0x00 0x3C 0x00 0xC0]
        OnnxIO.Proto.Writer tp = new OnnxIO.Proto.Writer();
        tp.varint(1, 2);
        tp.varint(2, 10);
        tp.string(8, "h");
        tp.bytes(9, new byte[] {0x00, 0x3C, 0x00, (byte) 0xC0});
        OnnxIO.Proto.Writer graph = new OnnxIO.Proto.Writer();
        graph.bytes(5, tp.toByteArray());
        OnnxIO.Proto.Writer model = new OnnxIO.Proto.Writer();
        model.bytes(7, graph.toByteArray());
        Path file = tmp.resolve("f16.onnx");
        Files.write(file, model.toByteArray());

        float[] got = OnnxIO.read(file).tensors().get("h").data();
        assertArrayEquals(new float[] {1f, -2f}, got, 0f);
    }

    @Test
    void fp16SpecialValues() {
        assertEquals(0f, DTypes.fp16ToFloat(0x0000));
        assertEquals(Float.POSITIVE_INFINITY, DTypes.fp16ToFloat(0x7c00));
        assertTrue(Float.isNaN(DTypes.fp16ToFloat(0x7e00)));
        // smallest positive subnormal: 2^-24
        assertEquals((float) Math.pow(2, -24), DTypes.fp16ToFloat(0x0001), 0f);
    }

    @Test
    void rejectsNonGgufFile() throws IOException {
        Path file = tmp.resolve("bogus.gguf");
        Files.writeString(file, "this is not a gguf file, just text padding to 24+ bytes");
        assertThrows(IOException.class, () -> GgufIO.read(file));
    }

    private static void writeLe(ByteArrayOutputStream o, long v, int bytes) {
        for (int i = 0; i < bytes; i++) o.write((int) (v >>> (8 * i)) & 0xff);
    }

    private static void writeStr(ByteArrayOutputStream o, String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        writeLe(o, b.length, 8);
        o.write(b, 0, b.length);
    }
}
