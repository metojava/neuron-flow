package data.model;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Decoding of little-endian tensor element types into float32. */
final class DTypes {

    private DTypes() {
    }

    /** Bytes per element for a supported dtype name, or -1 if unsupported. */
    static int elementSize(String dtype) {
        switch (dtype) {
            case "F32": return 4;
            case "F16":
            case "BF16": return 2;
            case "F64": return 8;
            default: return -1;
        }
    }

    /**
     * Decodes {@code length} bytes starting at {@code offset} into floats.
     *
     * @param dtype one of F32, F16, BF16, F64
     */
    static float[] decode(String dtype, byte[] src, int offset, int length) throws IOException {
        int size = elementSize(dtype);
        if (size < 0) {
            throw new IOException("Unsupported tensor dtype: " + dtype);
        }
        if (length % size != 0) {
            throw new IOException("Byte length " + length + " is not a multiple of " + dtype);
        }
        ByteBuffer bb = ByteBuffer.wrap(src, offset, length).order(ByteOrder.LITTLE_ENDIAN);
        int n = length / size;
        float[] out = new float[n];
        switch (dtype) {
            case "F32":
                bb.asFloatBuffer().get(out);
                break;
            case "F16":
                for (int i = 0; i < n; i++) out[i] = fp16ToFloat(bb.getShort() & 0xffff);
                break;
            case "BF16":
                for (int i = 0; i < n; i++) {
                    out[i] = Float.intBitsToFloat((bb.getShort() & 0xffff) << 16);
                }
                break;
            default: // F64
                for (int i = 0; i < n; i++) out[i] = (float) bb.getDouble();
                break;
        }
        return out;
    }

    /** IEEE 754 half precision to float, including subnormals, infinities and NaN. */
    static float fp16ToFloat(int h) {
        int sign = (h >>> 15) & 1;
        int exp = (h >>> 10) & 0x1f;
        int mant = h & 0x3ff;
        int bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign << 31;
            } else {
                int e = -1;
                do {
                    e++;
                    mant <<= 1;
                } while ((mant & 0x400) == 0);
                mant &= 0x3ff;
                bits = (sign << 31) | ((127 - 15 - e) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            bits = (sign << 31) | 0x7f800000 | (mant << 13);
        } else {
            bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        return Float.intBitsToFloat(bits);
    }

    /** Writes floats as little-endian float32 bytes. */
    static byte[] encodeF32(float[] data) {
        ByteBuffer bb = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bb.asFloatBuffer().put(data);
        return bb.array();
    }
}
