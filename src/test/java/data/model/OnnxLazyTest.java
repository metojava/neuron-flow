package data.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class OnnxLazyTest {

    @TempDir
    Path tmp;

    @Test
    void lazyReaderLoadsOnlyRequestedInitializers() throws IOException {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        tensors.put("w", new Tensor("w", new int[] {2, 2}, new float[] {1, 2, 3, 4}));
        tensors.put("b", new Tensor("b", new int[] {2}, new float[] {-1, 0.5f}));
        Path file = tmp.resolve("lazy.onnx");
        OnnxIO.write(file, new WeightFile(new LinkedHashMap<>(), tensors));

        try (OnnxIO.Reader r = new OnnxIO.Reader(file)) {
            assertEquals(2, r.names().size());
            Tensor b = r.load("b");
            assertArrayEquals(new float[] {-1, 0.5f}, b.data(), 0f);
            assertArrayEquals(new int[] {2, 2}, r.load("w").shape());
        }
    }

    @Test
    void packedFloatDataIsReadLazily() throws IOException {
        ByteBuffer packed = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        packed.putFloat(1.5f).putFloat(-2f);

        OnnxIO.Proto.Writer tp = new OnnxIO.Proto.Writer();
        tp.varint(1, 2);            // dims
        tp.varint(2, 1);            // FLOAT
        tp.string(8, "f");
        tp.bytes(4, packed.array()); // float_data (packed)
        OnnxIO.Proto.Writer graph = new OnnxIO.Proto.Writer();
        graph.bytes(5, tp.toByteArray());
        OnnxIO.Proto.Writer model = new OnnxIO.Proto.Writer();
        model.bytes(7, graph.toByteArray());
        Path file = tmp.resolve("floatdata.onnx");
        Files.write(file, model.toByteArray());

        try (OnnxIO.Reader r = new OnnxIO.Reader(file)) {
            assertArrayEquals(new float[] {1.5f, -2f}, r.load("f").data(), 0f);
        }
    }

    @Test
    void unsupportedDataTypeIsRejectedWhenOpened() throws IOException {
        OnnxIO.Proto.Writer tp = new OnnxIO.Proto.Writer();
        tp.varint(1, 1);
        tp.varint(2, 7);            // INT64: not a supported weight type
        tp.string(8, "i");
        tp.bytes(9, new byte[8]);
        OnnxIO.Proto.Writer graph = new OnnxIO.Proto.Writer();
        graph.bytes(5, tp.toByteArray());
        OnnxIO.Proto.Writer model = new OnnxIO.Proto.Writer();
        model.bytes(7, graph.toByteArray());
        Path file = tmp.resolve("int64.onnx");
        Files.write(file, model.toByteArray());

        IOException e = assertThrows(IOException.class, () -> new OnnxIO.Reader(file));
        assertTrue(e.getMessage().contains("unsupported ONNX data type"));
    }

    @Test
    void truncatedFileIsRejected() throws IOException {
        Path file = tmp.resolve("trunc.onnx");
        OnnxIO.write(file, new WeightFile(new LinkedHashMap<>(),
                Map.of("x", new Tensor("x", new int[] {3}, new float[] {1, 2, 3}))));
        byte[] all = Files.readAllBytes(file);
        Files.write(file, java.util.Arrays.copyOf(all, all.length - 4));
        assertThrows(IOException.class, () -> new OnnxIO.Reader(file));
    }
}
