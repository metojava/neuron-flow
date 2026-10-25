package data.io.parquet;

import java.io.IOException;

/**
 * Run-length / bit-packing hybrid encoding used for Parquet levels, boolean
 * values and dictionary indices.
 */
final class Rle {

    private Rle() {
    }

    /**
     * Encodes the values as one bit-packed run. No length prefix and no bit
     * width byte is written: callers add those where the format requires them.
     */
    static byte[] encodeBitPacked(int[] values, int bitWidth) {
        if (values.length == 0) {
            return new byte[0];
        }
        int groups = (values.length + 7) / 8;
        int byteCount = groups * bitWidth;
        byte[] out = new byte[byteCount];
        int padded = groups * 8;
        for (int i = 0; i < padded; i++) {
            int value = i < values.length ? values[i] : 0;
            for (int bit = 0; bit < bitWidth; bit++) {
                if (((value >>> bit) & 1) != 0) {
                    long pos = (long) i * bitWidth + bit;
                    out[(int) (pos >>> 3)] |= (byte) (1 << (pos & 7));
                }
            }
        }
        Writer header = new Writer();
        header.varint(((long) groups << 1) | 1);
        byte[] headerBytes = header.toByteArray();
        byte[] result = new byte[headerBytes.length + out.length];
        System.arraycopy(headerBytes, 0, result, 0, headerBytes.length);
        System.arraycopy(out, 0, result, headerBytes.length, out.length);
        return result;
    }

    /**
     * Decodes exactly {@code count} values from a run-length / bit-packed
     * hybrid stream.
     *
     * @param data   backing array
     * @param offset start of the encoded stream
     * @param length number of readable bytes
     * @param bitWidth width of each value in bits (0 to 32)
     * @param count  number of values to return
     */
    static int[] decode(byte[] data, int offset, int length, int bitWidth, int count) throws IOException {
        if (bitWidth < 0 || bitWidth > 32) {
            throw new IOException("Invalid RLE bit width " + bitWidth);
        }
        int[] out = new int[count];
        if (count == 0) {
            return out;
        }
        Cursor cursor = new Cursor(data, offset, offset + length);
        int produced = 0;
        while (produced < count) {
            long header = cursor.varint();
            if ((header & 1) != 0) {
                long groups = header >>> 1;
                long valuesInRun = groups * 8;
                long bytes = groups * bitWidth;
                if (groups <= 0 || bytes < 0 || bytes > cursor.remaining()) {
                    throw new IOException("Corrupt bit-packed run in Parquet data");
                }
                int base = cursor.position();
                int take = (int) Math.min(valuesInRun, count - produced);
                for (int i = 0; i < take; i++) {
                    out[produced + i] = unpack(data, base, (long) i * bitWidth, bitWidth);
                }
                produced += take;
                cursor.skip((int) bytes);
            } else {
                long runLength = header >>> 1;
                if (runLength <= 0) {
                    throw new IOException("Corrupt RLE run in Parquet data");
                }
                int value = readRunValue(cursor, bitWidth);
                int take = (int) Math.min(runLength, count - produced);
                for (int i = 0; i < take; i++) {
                    out[produced + i] = value;
                }
                produced += take;
            }
        }
        return out;
    }

    static int unpack(byte[] data, int base, long bitPosition, int bitWidth) {
        if (bitWidth == 0) {
            return 0;
        }
        int value = 0;
        int produced = 0;
        int remaining = bitWidth;
        long bit = bitPosition;
        while (remaining > 0) {
            int byteIndex = base + (int) (bit >>> 3);
            int shift = (int) (bit & 7);
            int bits = Math.min(8 - shift, remaining);
            int mask = (1 << bits) - 1;
            value |= ((data[byteIndex] >>> shift) & mask) << produced;
            produced += bits;
            remaining -= bits;
            bit += bits;
        }
        return value;
    }

    private static int readRunValue(Cursor cursor, int bitWidth) throws IOException {
        int byteCount = (bitWidth + 7) / 8;
        if (byteCount == 0) {
            return 0;
        }
        if (byteCount > cursor.remaining()) {
            throw new IOException("Corrupt RLE value in Parquet data");
        }
        int value = 0;
        for (int i = 0; i < byteCount; i++) {
            value |= (cursor.readByte() & 0xff) << (8 * i);
        }
        return value;
    }

    private static final class Cursor {
        private final byte[] data;
        private final int end;
        private int position;

        Cursor(byte[] data, int offset, int end) {
            this.data = data;
            this.position = offset;
            this.end = end;
        }

        int position() {
            return position;
        }

        int remaining() {
            return end - position;
        }

        int readByte() throws IOException {
            if (position >= end) {
                throw new IOException("Unexpected end of RLE data");
            }
            return data[position++] & 0xff;
        }

        void skip(int count) throws IOException {
            if (count < 0 || count > remaining()) {
                throw new IOException("Unexpected end of RLE data");
            }
            position += count;
        }

        long varint() throws IOException {
            long result = 0;
            int shift = 0;
            while (true) {
                int b = readByte();
                result |= (long) (b & 0x7f) << shift;
                if ((b & 0x80) == 0) {
                    return result;
                }
                shift += 7;
                if (shift > 63) {
                    throw new IOException("RLE varint is too long");
                }
            }
        }
    }

    private static final class Writer {
        private byte[] buffer = new byte[8];
        private int size;

        void varint(long value) {
            while ((value & ~0x7fL) != 0) {
                writeByte((int) ((value & 0x7f) | 0x80));
                value >>>= 7;
            }
            writeByte((int) value);
        }

        byte[] toByteArray() {
            byte[] result = new byte[size];
            System.arraycopy(buffer, 0, result, 0, size);
            return result;
        }

        private void writeByte(int value) {
            if (size == buffer.length) {
                buffer = java.util.Arrays.copyOf(buffer, buffer.length * 2);
            }
            buffer[size++] = (byte) value;
        }
    }
}
