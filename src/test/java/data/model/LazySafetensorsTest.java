package data.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class LazySafetensorsTest {

    @TempDir
    Path tmp;

    @Test
    void readsEachTensorOnDemand() throws IOException {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        tensors.put("a", new Tensor("a", new int[] {2, 2}, new float[] {1, 2, 3, 4}));
        tensors.put("b", new Tensor("b", new int[] {3}, new float[] {-1, 0, 0.5f}));
        Map<String, String> meta = new LinkedHashMap<>();
        meta.put("source", "test");
        Path file = tmp.resolve("lazy.safetensors");
        SafetensorsIO.write(file, new WeightFile(meta, tensors));

        try (SafetensorsIO.Reader r = new SafetensorsIO.Reader(file)) {
            assertEquals(2, r.names().size());
            assertTrue(r.names().contains("a"));
            assertEquals("test", r.metadata().get("source"));

            Tensor b = r.load("b");
            assertArrayEquals(new int[] {3}, b.shape());
            assertArrayEquals(new float[] {-1, 0, 0.5f}, b.data(), 0f);

            Tensor a = r.load("a");
            assertArrayEquals(new float[] {1, 2, 3, 4}, a.data(), 0f);
        }
    }

    @Test
    void unknownNameIsRejected() throws IOException {
        Path file = tmp.resolve("one.safetensors");
        SafetensorsIO.write(file, new WeightFile(new LinkedHashMap<>(),
                Map.of("x", new Tensor("x", new int[] {1}, new float[] {1f}))));
        try (SafetensorsIO.Reader r = new SafetensorsIO.Reader(file)) {
            assertThrows(IllegalArgumentException.class, () -> r.load("missing"));
        }
    }

    @Test
    void truncatedFileIsRejectedOnOpen() throws IOException {
        Path file = tmp.resolve("trunc.safetensors");
        SafetensorsIO.write(file, new WeightFile(new LinkedHashMap<>(),
                Map.of("x", new Tensor("x", new int[] {4}, new float[] {1, 2, 3, 4}))));
        byte[] all = Files.readAllBytes(file);
        Files.write(file, java.util.Arrays.copyOf(all, all.length - 8));
        assertThrows(IOException.class, () -> new SafetensorsIO.Reader(file));
    }
}
