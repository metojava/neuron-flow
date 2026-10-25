package data.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class StreamingWriteTest {

    @TempDir
    Path tmp;

    private static Map<String, Tensor> sourceTensors(int count, int size, long seed) {
        Random r = new Random(seed);
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (int k = 0; k < count; k++) {
            float[] v = new float[size];
            for (int i = 0; i < size; i++) v[i] = (float) r.nextGaussian();
            String name = "t" + k;
            out.put(name, new Tensor(name, new int[] {size / 100, 100}, v));
        }
        return out;
    }

    private static Map<String, int[]> shapesOf(Map<String, Tensor> tensors) {
        Map<String, int[]> shapes = new LinkedHashMap<>();
        for (Tensor t : tensors.values()) shapes.put(t.name(), t.shape());
        return shapes;
    }

    @Test
    void safetensorsCopiedTensorByTensorFromLazyReader() throws IOException {
        Map<String, Tensor> src = sourceTensors(6, 100_000, 1L);
        Path in = tmp.resolve("in.safetensors");
        SafetensorsIO.write(in, new WeightFile(new LinkedHashMap<>(), src));

        Path out = tmp.resolve("out.safetensors");
        Map<String, String> meta = new LinkedHashMap<>();
        meta.put("copied", "yes");
        try (SafetensorsIO.Reader r = new SafetensorsIO.Reader(in);
             SafetensorsIO.StreamWriter w = new SafetensorsIO.StreamWriter(out, shapesOf(src), meta)) {
            for (String name : src.keySet()) {
                Tensor t = r.load(name);          // one tensor in memory at a time
                w.write(t.name(), t.data());
            }
            assertEquals(6, w.tensorsWritten());
        }

        WeightFile back = SafetensorsIO.read(out);
        assertEquals("yes", back.metadata().get("copied"));
        for (Tensor t : src.values()) {
            assertArrayEquals(t.data(), back.tensors().get(t.name()).data(), 0f);
        }
    }

    @Test
    void ggufCopiedTensorByTensorFromLazyReader() throws IOException {
        Map<String, Tensor> src = sourceTensors(4, 50_000, 2L);
        Path in = tmp.resolve("in.gguf");
        GgufIO.write(in, new WeightFile(new LinkedHashMap<>(), src));

        Path out = tmp.resolve("out.gguf");
        try (GgufIO.Reader r = new GgufIO.Reader(in);
             GgufIO.StreamWriter w = new GgufIO.StreamWriter(out, shapesOf(src), new LinkedHashMap<>())) {
            for (String name : src.keySet()) {
                Tensor t = r.load(name);
                w.write(t.name(), t.data());
            }
        }

        WeightFile back = GgufIO.read(out);
        for (Tensor t : src.values()) {
            assertArrayEquals(t.data(), back.tensors().get(t.name()).data(), 0f);
        }
    }

    @Test
    void outOfOrderNameIsRejected() throws IOException {
        Map<String, int[]> shapes = new LinkedHashMap<>();
        shapes.put("a", new int[] {2});
        shapes.put("b", new int[] {2});
        SafetensorsIO.StreamWriter w = new SafetensorsIO.StreamWriter(
                tmp.resolve("order.safetensors"), shapes, null);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> w.write("b", new float[] {1, 2}));
        assertTrue(e.getMessage().contains("Expected tensor a"));
        // Nothing valid was written, so closing reports the incomplete file.
        assertThrows(IOException.class, w::close);
    }

    @Test
    void wrongLengthIsRejected() throws IOException {
        Map<String, int[]> shapes = new LinkedHashMap<>();
        shapes.put("a", new int[] {2, 2});
        GgufIO.StreamWriter w = new GgufIO.StreamWriter(tmp.resolve("len.gguf"), shapes, null);
        assertThrows(IllegalArgumentException.class, () -> w.write("a", new float[] {1, 2, 3}));
        assertThrows(IOException.class, w::close);
    }

    @Test
    void closingEarlyDeletesThePartialFile() throws IOException {
        Map<String, int[]> shapes = new LinkedHashMap<>();
        shapes.put("a", new int[] {1});
        shapes.put("b", new int[] {1});
        Path file = tmp.resolve("partial.safetensors");
        SafetensorsIO.StreamWriter w = new SafetensorsIO.StreamWriter(file, shapes, null);
        w.write("a", new float[] {1f});
        IOException e = assertThrows(IOException.class, w::close);
        assertTrue(e.getMessage().contains("wrote 1 of 2"));
        assertFalse(Files.exists(file));
    }

    private static void assertTrue(boolean condition) {
        org.junit.jupiter.api.Assertions.assertTrue(condition);
    }
}
