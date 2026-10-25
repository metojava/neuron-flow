package data.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

/** Hand-built K-quant super-blocks checked against ggml's dequantization formulas. */
class GgufKQuantTest {

    private static final short ONE = 0x3C00;
    private static final short HALF = 0x3800;

    private static ByteBuffer block(int bytes) {
        return ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    @Test
    void byteCountUsesSuperBlocksOf256() throws IOException {
        assertEquals(144L, GgufQuant.byteCount(GgufQuant.Q4_K, 256));
        assertEquals(420L, GgufQuant.byteCount(GgufQuant.Q6_K, 512));
        assertThrows(IOException.class, () -> GgufQuant.byteCount(GgufQuant.Q4_K, 128));
    }

    @Test
    void q2_KUsesFourBitScaleAndMin() throws IOException {
        ByteBuffer b = block(84);
        b.put(0, (byte) 0x21);    // scale 1, min 2 for sub-block 0
        b.putShort(80, ONE);      // d
        b.putShort(82, HALF);     // dmin
        b.put(16, (byte) 2);      // quant 2 at shift 0
        float[] v = GgufQuant.dequantize(GgufQuant.Q2_K, b.array(), 0, 256);
        assertEquals(1f, v[0], 1e-6f);   // 1 * 2 - 0.5 * 2
    }

    @Test
    void q3_KZeroScalesGiveScaleMinus32() throws IOException {
        ByteBuffer b = block(110);
        b.putShort(108, ONE);     // d; packed scales are all zero -> scale = -32
        float[] v = GgufQuant.dequantize(GgufQuant.Q3_K, b.array(), 0, 256);
        // high bit clear -> value (0 - 4) times scale -32
        assertEquals(128f, v[0], 1e-6f);
    }

    @Test
    void q4_KScalesAndMinsSelectSubBlocks() throws IOException {
        ByteBuffer b = block(144);
        b.putShort(0, ONE);       // d
        b.putShort(2, HALF);      // dmin
        // Scales of sub-blocks 0-3 live in bytes 0-3 of the packed array; minimums are zero.
        // Sub-blocks 4-7 take their scale bits from bytes 8-11, which stay zero here.
        for (int j = 0; j < 4; j++) b.put(4 + j, (byte) 1);
        for (int i = 0; i < 128; i++) b.put(16 + i, (byte) 0x22); // every nibble = 2
        float[] v = GgufQuant.dequantize(GgufQuant.Q4_K, b.array(), 0, 256);
        assertEquals(2f, v[0], 1e-6f);    // scale 1 * nibble 2
        assertEquals(2f, v[31], 1e-6f);
        assertEquals(0f, v[128], 1e-6f);  // sub-blocks 4-7 have scale 0
    }

    @Test
    void q5_KHighBitAddsSixteen() throws IOException {
        ByteBuffer b = block(176);
        b.putShort(0, ONE);
        for (int j = 0; j < 4; j++) b.put(4 + j, (byte) 1);
        b.put(16, (byte) 0x01);   // fifth bit for element 0
        float[] v = GgufQuant.dequantize(GgufQuant.Q5_K, b.array(), 0, 256);
        assertEquals(16f, v[0], 1e-6f);
        assertEquals(0f, v[1], 1e-6f);
    }

    @Test
    void q6_KUsesSignedScalesAndCentredValues() throws IOException {
        ByteBuffer b = block(210);
        for (int i = 0; i < 16; i++) b.put(192 + i, (byte) 1);
        b.putShort(208, ONE);
        float[] v = GgufQuant.dequantize(GgufQuant.Q6_K, b.array(), 0, 256);
        // ql = qh = 0 -> 6-bit value 0 - 32 = -32, scale 1
        assertEquals(-32f, v[0], 1e-6f);
        assertEquals(-32f, v[200], 1e-6f);
    }

    @Test
    void q8_KScalesInt8ByFloatScale() throws IOException {
        ByteBuffer b = block(292);
        b.putFloat(0, 0.5f);
        b.put(4, (byte) -4);
        b.put(4 + 255, (byte) 10);
        float[] v = GgufQuant.dequantize(GgufQuant.Q8_K, b.array(), 0, 256);
        assertEquals(-2f, v[0], 1e-6f);
        assertEquals(5f, v[255], 1e-6f);
    }
}
