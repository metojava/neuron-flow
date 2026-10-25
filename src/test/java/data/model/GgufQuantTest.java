package data.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GgufQuantTest {

    @TempDir
    Path tmp;

    // IEEE half-precision bit patterns
    private static final short ONE = 0x3C00;
    private static final short TWO = 0x4000;
    private static final short THREE = 0x4200;
    private static final short HALF = 0x3800;
    private static final short QUARTER = 0x3400;

    private static ByteBuffer block(int bytes) {
        return ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    @Test
    void q4_0NibblesAreCenteredAtZero() throws IOException {
        ByteBuffer b = block(18);
        b.putShort(ONE);
        b.put(new byte[16]);
        b.put(2, (byte) 0x21);   // low nibble 1, high nibble 2
        b.put(3, (byte) 0xF0);   // low nibble 0, high nibble 15
        float[] v = GgufQuant.dequantize(GgufQuant.Q4_0, b.array(), 0, 32);
        assertEquals(-7f, v[0]);
        assertEquals(-6f, v[16]);
        assertEquals(-8f, v[1]);
        assertEquals(7f, v[17]);
    }

    @Test
    void q4_1UsesScaleAndMinimum() throws IOException {
        ByteBuffer b = block(20);
        b.putShort(TWO).putShort(THREE);
        b.put(4, (byte) 0x21);
        float[] v = GgufQuant.dequantize(GgufQuant.Q4_1, b.array(), 0, 32);
        assertEquals(5f, v[0]);   // 1 * 2 + 3
        assertEquals(7f, v[16]);  // 2 * 2 + 3
    }

    @Test
    void q5_0FifthBitComesFromHeader() throws IOException {
        ByteBuffer b = block(22);
        b.putShort(ONE);
        b.putInt(2, 1 << 16);     // fifth bit of the high-half value 0 (bit 16 + j)
        float[] v = GgufQuant.dequantize(GgufQuant.Q5_0, b.array(), 0, 32);
        assertEquals(-16f, v[0]);  // low half: 0 - 16, no fifth bit
        assertEquals(0f, v[16]);   // high half: 0 + 16 - 16
    }

    @Test
    void q5_1AddsMinimumToUnsignedFifthBitValues() throws IOException {
        ByteBuffer b = block(24);
        b.putShort(ONE).putShort(HALF);
        b.put(8, (byte) 0x05);
        float[] v = GgufQuant.dequantize(GgufQuant.Q5_1, b.array(), 0, 32);
        assertEquals(5.5f, v[0]);
        assertEquals(0.5f, v[16]);
    }

    @Test
    void q8_0ScalesSignedBytes() throws IOException {
        ByteBuffer b = block(34);
        b.putShort(HALF);
        b.put(2, (byte) -4);
        b.put(33, (byte) 10);
        float[] v = GgufQuant.dequantize(GgufQuant.Q8_0, b.array(), 0, 32);
        assertEquals(-2f, v[0]);
        assertEquals(5f, v[31]);
    }

    @Test
    void q8_1IgnoresSumField() throws IOException {
        ByteBuffer b = block(36);
        b.putShort(QUARTER).putShort((short) 0x7000); // second half is the sum, ignored
        b.put(4, (byte) -4);
        b.put(35, (byte) 8);
        float[] v = GgufQuant.dequantize(GgufQuant.Q8_1, b.array(), 0, 32);
        assertEquals(-1f, v[0]);
        assertEquals(2f, v[31]);
    }

    @Test
    void byteCountFollowsBlockLayout() throws IOException {
        assertEquals(36L, GgufQuant.byteCount(GgufQuant.Q4_0, 64));
        assertEquals(72L, GgufQuant.byteCount(GgufQuant.Q8_1, 64));
        assertEquals(128L, GgufQuant.byteCount(GgufQuant.F32, 32));
        assertThrows(IOException.class, () -> GgufQuant.byteCount(GgufQuant.Q8_0, 33));
    }

    @Test
    void iqTypesAreRejected() {
        assertThrows(IOException.class, () -> GgufQuant.byteCount(16, 256));
    }

    @Test
    void quantizedTensorReadsFromFile() throws IOException {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        writeLe(o, 0x46554747, 4);
        writeLe(o, 3, 4);
        writeLe(o, 1, 8);   // one tensor
        writeLe(o, 0, 8);   // no metadata
        byte[] name = "w".getBytes(StandardCharsets.UTF_8);
        writeLe(o, name.length, 8);
        o.write(name, 0, name.length);
        writeLe(o, 1, 4);   // rank 1
        writeLe(o, 32, 8); // 32 values
        writeLe(o, 8, 4);   // Q8_0
        writeLe(o, 0, 8);   // offset within data section
        while (o.size() % 32 != 0) o.write(0);

        ByteBuffer block = block(34);
        block.putShort(HALF);
        block.put(2, (byte) -4);
        block.put(33, (byte) 10);
        o.write(block.array(), 0, 34);

        Path file = tmp.resolve("q8.gguf");
        Files.write(file, o.toByteArray());

        float[] eager = GgufIO.read(file).tensors().get("w").data();
        assertEquals(-2f, eager[0]);
        assertEquals(5f, eager[31]);

        try (GgufIO.Reader r = new GgufIO.Reader(file)) {
            assertEquals(1, r.names().size());
            assertArrayEquals(eager, r.load("w").data(), 0f);
        }
    }

    private static void writeLe(ByteArrayOutputStream o, long v, int bytes) {
        for (int i = 0; i < bytes; i++) o.write((int) (v >>> (8 * i)) & 0xff);
    }
}
