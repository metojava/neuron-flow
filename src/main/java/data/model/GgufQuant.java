package data.model;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * GGML tensor types and their dequantization to float32.
 *
 * <p>Supported types, with layouts taken from ggml's block structs and dequantization
 * routines:
 * <ul>
 *   <li>Plain: F32, F16, BF16.</li>
 *   <li>Legacy blocks of 32 values: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1.</li>
 *   <li>K-quant super-blocks of 256 values: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K.</li>
 * </ul>
 * IQ types and ternary types are not supported.
 */
final class GgufQuant {

    static final int F32 = 0;
    static final int F16 = 1;
    static final int Q4_0 = 2;
    static final int Q4_1 = 3;
    static final int Q5_0 = 6;
    static final int Q5_1 = 7;
    static final int Q8_0 = 8;
    static final int Q8_1 = 9;
    static final int Q2_K = 10;
    static final int Q3_K = 11;
    static final int Q4_K = 12;
    static final int Q5_K = 13;
    static final int Q6_K = 14;
    static final int Q8_K = 15;
    static final int BF16 = 30;

    private static final int BLOCK = 32;
    private static final int SUPER = 256;

    private GgufQuant() {
    }

    /** @return the dtype name used by {@link DTypes}, or null if the type is not a plain float type */
    static String plainName(int type) {
        switch (type) {
            case F32: return "F32";
            case F16: return "F16";
            case BF16: return "BF16";
            default: return null;
        }
    }

    /** @return true if this GGML type can be read */
    static boolean isSupported(int type) {
        return plainName(type) != null || blockBytes(type) > 0;
    }

    /** Values per block: 32 for legacy types, 256 for K-quants. */
    static int blockElems(int type) {
        return type >= Q2_K && type <= Q8_K ? SUPER : BLOCK;
    }

    /** Bytes per block for quantized types; 0 for plain and unsupported types. */
    static int blockBytes(int type) {
        switch (type) {
            case Q4_0: return 18;
            case Q4_1: return 20;
            case Q5_0: return 22;
            case Q5_1: return 24;
            case Q8_0: return 34;
            case Q8_1: return 36;
            case Q2_K: return 84;
            case Q3_K: return 110;
            case Q4_K: return 144;
            case Q5_K: return 176;
            case Q6_K: return 210;
            case Q8_K: return 292;
            default: return 0;
        }
    }

    /**
     * Number of stored bytes for {@code numel} values of the given type.
     *
     * @throws IOException if the type is unsupported or numel does not fill whole blocks
     */
    static long byteCount(int type, long numel) throws IOException {
        String plain = plainName(type);
        if (plain != null) {
            return numel * DTypes.elementSize(plain);
        }
        int bb = blockBytes(type);
        if (bb == 0) {
            throw unsupported(type);
        }
        int elems = blockElems(type);
        if (numel % elems != 0) {
            throw new IOException("Tensor size " + numel + " is not a multiple of " + elems
                    + " for GGML type " + type);
        }
        return numel / elems * bb;
    }

    static IOException unsupported(int type) {
        return new IOException("Unsupported GGML tensor type " + type
                + " (supported: F32, F16, BF16, Q4_0-Q8_1, Q2_K-Q8_K; IQ and ternary types are not)");
    }

    /**
     * Decodes {@code numel} values starting at {@code offset} to float32.
     */
    static float[] dequantize(int type, byte[] src, int offset, int numel) throws IOException {
        String plain = plainName(type);
        if (plain != null) {
            return DTypes.decode(plain, src, offset, numel * DTypes.elementSize(plain));
        }
        if (blockBytes(type) == 0) {
            throw unsupported(type);
        }
        int elems = blockElems(type);
        int blocks = numel / elems;
        int bytes = blockBytes(type);
        float[] out = new float[numel];
        ByteBuffer bb = ByteBuffer.wrap(src).order(ByteOrder.LITTLE_ENDIAN);
        for (int b = 0; b < blocks; b++) {
            int base = offset + b * bytes;
            int o = b * elems;
            switch (type) {
                case Q4_0: dequantQ4_0(bb, src, base, out, o); break;
                case Q4_1: dequantQ4_1(bb, src, base, out, o); break;
                case Q5_0: dequantQ5_0(bb, src, base, out, o); break;
                case Q5_1: dequantQ5_1(bb, src, base, out, o); break;
                case Q8_0: dequantQ8_0(bb, src, base, out, o); break;
                case Q8_1: dequantQ8_1(bb, src, base, out, o); break;
                case Q2_K: dequantQ2_K(bb, src, base, out, o); break;
                case Q3_K: dequantQ3_K(bb, src, base, out, o); break;
                case Q4_K: dequantQ4_K(bb, src, base, out, o); break;
                case Q5_K: dequantQ5_K(bb, src, base, out, o); break;
                case Q6_K: dequantQ6_K(bb, src, base, out, o); break;
                default: dequantQ8_K(bb, src, base, out, o); break;
            }
        }
        return out;
    }

    // ---- legacy 32-value blocks ----

    private static void dequantQ4_0(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float d = half(bb, base);
        for (int j = 0; j < 16; j++) {
            int q = u8(s, base + 2 + j);
            out[o + j] = ((q & 0xF) - 8) * d;
            out[o + j + 16] = ((q >> 4) - 8) * d;
        }
    }

    private static void dequantQ4_1(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float d = half(bb, base);
        float m = half(bb, base + 2);
        for (int j = 0; j < 16; j++) {
            int q = u8(s, base + 4 + j);
            out[o + j] = (q & 0xF) * d + m;
            out[o + j + 16] = (q >> 4) * d + m;
        }
    }

    private static void dequantQ5_0(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float d = half(bb, base);
        int qh = bb.getInt(base + 2);
        for (int j = 0; j < 16; j++) {
            int q = u8(s, base + 6 + j);
            int xh0 = ((qh >>> j) << 4) & 0x10;
            int xh1 = (qh >>> (j + 12)) & 0x10;
            out[o + j] = (((q & 0xF) | xh0) - 16) * d;
            out[o + j + 16] = (((q >> 4) | xh1) - 16) * d;
        }
    }

    private static void dequantQ5_1(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float d = half(bb, base);
        float m = half(bb, base + 2);
        int qh = bb.getInt(base + 4);
        for (int j = 0; j < 16; j++) {
            int q = u8(s, base + 8 + j);
            int xh0 = ((qh >>> j) << 4) & 0x10;
            int xh1 = (qh >>> (j + 12)) & 0x10;
            out[o + j] = ((q & 0xF) | xh0) * d + m;
            out[o + j + 16] = ((q >> 4) | xh1) * d + m;
        }
    }

    private static void dequantQ8_0(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float d = half(bb, base);
        for (int j = 0; j < BLOCK; j++) {
            out[o + j] = s[base + 2 + j] * d;
        }
    }

    private static void dequantQ8_1(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float d = half(bb, base); // second half-precision field is the sum, not needed
        for (int j = 0; j < BLOCK; j++) {
            out[o + j] = s[base + 4 + j] * d;
        }
    }

    // ---- K-quant 256-value super-blocks ----

    /** Q2_K: scales[16] (4-bit scale, 4-bit min), qs[64], d, dmin. */
    private static void dequantQ2_K(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float d = half(bb, base + 80);
        float min = half(bb, base + 82);
        int is = 0;
        int yo = o;
        int q = base + 16;
        for (int n = 0; n < SUPER; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int sc = u8(s, base + is++);
                float dl = d * (sc & 0xF);
                float ml = min * (sc >> 4);
                for (int l = 0; l < 16; l++) {
                    out[yo++] = dl * ((u8(s, q + l) >> shift) & 3) - ml;
                }
                sc = u8(s, base + is++);
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16; l++) {
                    out[yo++] = dl * ((u8(s, q + 16 + l) >> shift) & 3) - ml;
                }
                shift += 2;
            }
            q += 32;
        }
    }

    /** Q3_K: hmask[32], qs[64], scales[12] (6-bit, packed), d. */
    private static void dequantQ3_K(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float dAll = half(bb, base + 108);
        final long kmask1 = 0x03030303L;
        final long kmask2 = 0x0f0f0f0fL;
        long a0 = u32(bb, base + 96);
        long a1 = u32(bb, base + 100);
        long tmp = u32(bb, base + 104);
        long aux2 = ((a0 >>> 4) & kmask2) | (((tmp >>> 4) & kmask1) << 4);
        long aux3 = ((a1 >>> 4) & kmask2) | (((tmp >>> 6) & kmask1) << 4);
        long aux0 = (a0 & kmask2) | (((tmp >>> 0) & kmask1) << 4);
        long aux1 = (a1 & kmask2) | (((tmp >>> 2) & kmask1) << 4);
        long[] aux = {aux0, aux1, aux2, aux3};
        int[] scales = new int[16];
        for (int k = 0; k < 4; k++) {
            for (int b = 0; b < 4; b++) {
                scales[k * 4 + b] = (byte) ((aux[k] >>> (8 * b)) & 0xff);
            }
        }

        int is = 0;
        int yo = o;
        int q = base + 32;
        int hm = base;
        int m = 1;
        for (int n = 0; n < SUPER; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = dAll * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int high = (u8(s, hm + l) & m) != 0 ? 0 : 4;
                    out[yo++] = dl * (((u8(s, q + l) >> shift) & 3) - high);
                }
                dl = dAll * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int high = (u8(s, hm + 16 + l) & m) != 0 ? 0 : 4;
                    out[yo++] = dl * (((u8(s, q + 16 + l) >> shift) & 3) - high);
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }

    /** Q4_K: d, dmin, scales[12] (6-bit scales and mins), qs[128]. */
    private static void dequantQ4_K(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float d = half(bb, base);
        float min = half(bb, base + 2);
        int sc0 = base + 4;
        int q = base + 16;
        int yo = o;
        int is = 0;
        for (int j = 0; j < SUPER; j += 64) {
            int sc = scaleOf(s, sc0, is);
            int mn = minOf(s, sc0, is);
            float d1 = d * sc;
            float m1 = min * mn;
            sc = scaleOf(s, sc0, is + 1);
            mn = minOf(s, sc0, is + 1);
            float d2 = d * sc;
            float m2 = min * mn;
            for (int l = 0; l < 32; l++) out[yo++] = d1 * (u8(s, q + l) & 0xF) - m1;
            for (int l = 0; l < 32; l++) out[yo++] = d2 * (u8(s, q + l) >> 4) - m2;
            q += 32;
            is += 2;
        }
    }

    /** Q5_K: d, dmin, scales[12], qh[32] (high bits), qs[128] (low nibbles). */
    private static void dequantQ5_K(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float d = half(bb, base);
        float min = half(bb, base + 2);
        int sc0 = base + 4;
        int qh = base + 16;
        int ql = base + 48;
        int yo = o;
        int is = 0;
        int u1 = 1;
        int u2 = 2;
        for (int j = 0; j < SUPER; j += 64) {
            int sc = scaleOf(s, sc0, is);
            int mn = minOf(s, sc0, is);
            float d1 = d * sc;
            float m1 = min * mn;
            sc = scaleOf(s, sc0, is + 1);
            mn = minOf(s, sc0, is + 1);
            float d2 = d * sc;
            float m2 = min * mn;
            for (int l = 0; l < 32; l++) {
                int hi = (u8(s, qh + l) & u1) != 0 ? 16 : 0;
                out[yo++] = d1 * ((u8(s, ql + l) & 0xF) + hi) - m1;
            }
            for (int l = 0; l < 32; l++) {
                int hi = (u8(s, qh + l) & u2) != 0 ? 16 : 0;
                out[yo++] = d2 * ((u8(s, ql + l) >> 4) + hi) - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    /** Q6_K: ql[128], qh[64], scales[16] (signed 8-bit), d. */
    private static void dequantQ6_K(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float d = half(bb, base + 208);
        int ql = base;
        int qh = base + 128;
        int sc = base + 192;
        int yo = o;
        for (int n = 0; n < SUPER; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = ((u8(s, ql + l) & 0xF) | (((u8(s, qh + l) >> 0) & 3) << 4)) - 32;
                int q2 = ((u8(s, ql + l + 32) & 0xF) | (((u8(s, qh + l) >> 2) & 3) << 4)) - 32;
                int q3 = ((u8(s, ql + l) >> 4) | (((u8(s, qh + l) >> 4) & 3) << 4)) - 32;
                int q4 = ((u8(s, ql + l + 32) >> 4) | (((u8(s, qh + l) >> 6) & 3) << 4)) - 32;
                out[yo + l] = d * s[sc + is] * q1;
                out[yo + l + 32] = d * s[sc + is + 2] * q2;
                out[yo + l + 64] = d * s[sc + is + 4] * q3;
                out[yo + l + 96] = d * s[sc + is + 6] * q4;
            }
            yo += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

    /** Q8_K: float d, qs[256] int8, bsums (not needed for dequantization). */
    private static void dequantQ8_K(ByteBuffer bb, byte[] s, int base, float[] out, int o) {
        float d = bb.getFloat(base);
        for (int j = 0; j < SUPER; j++) {
            out[o + j] = s[base + 4 + j] * d;
        }
    }

    // ---- helpers ----

    /** Scale for sub-block j from a 12-byte packed scales array at {@code at}. */
    private static int scaleOf(byte[] s, int at, int j) {
        if (j < 4) {
            return u8(s, at + j) & 63;
        }
        return (u8(s, at + j + 4) & 0xF) | ((u8(s, at + j - 4) >> 6) << 4);
    }

    /** Minimum for sub-block j from a 12-byte packed scales array at {@code at}. */
    private static int minOf(byte[] s, int at, int j) {
        if (j < 4) {
            return u8(s, at + j + 4) & 63;
        }
        return (u8(s, at + j + 4) >> 4) | ((u8(s, at + j) >> 6) << 4);
    }

    private static int u8(byte[] s, int i) {
        return s[i] & 0xff;
    }

    private static long u32(ByteBuffer bb, int at) {
        return bb.getInt(at) & 0xffffffffL;
    }

    private static float half(ByteBuffer bb, int at) {
        return DTypes.fp16ToFloat(bb.getShort(at) & 0xffff);
    }
}
