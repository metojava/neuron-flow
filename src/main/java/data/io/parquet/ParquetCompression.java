package data.io.parquet;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

/**
 * Pure-Java decompression for the Parquet codecs that can be implemented with
 * the JDK alone. Compressed pages are decompressed in memory.
 */
final class ParquetCompression {

    private ParquetCompression() {
    }

    static byte[] decompress(int codec, byte[] data, int offset, int length, int expectedSize)
            throws IOException {
        if (codec == ParquetFormat.UNCOMPRESSED) {
            return slice(data, offset, length);
        }
        return switch (codec) {
            case ParquetFormat.SNAPPY -> snappyDecompress(data, offset, length, expectedSize);
            case ParquetFormat.GZIP -> gzipDecompress(data, offset, length, expectedSize);
            case ParquetFormat.LZ4_RAW -> lz4RawDecompress(data, offset, length, expectedSize);
            default -> throw new IOException("Unsupported Parquet compression codec "
                    + ParquetFormat.codecName(codec) + " (supported: UNCOMPRESSED, SNAPPY, GZIP, LZ4_RAW)");
        };
    }

    static byte[] snappyDecompress(byte[] data, int offset, int length, int expectedSize)
            throws IOException {
        Cursor cursor = new Cursor(data, offset, offset + length);
        long declaredLong = cursor.varint();
        if (declaredLong < 0 || declaredLong > Integer.MAX_VALUE) {
            throw new IOException("Invalid Snappy uncompressed length " + declaredLong);
        }
        int declared = (int) declaredLong;
        if (expectedSize >= 0 && declared != expectedSize) {
            throw new IOException("Snappy page length mismatch: expected " + expectedSize
                    + " but compressed block declares " + declared);
        }
        byte[] out = new byte[declared];
        int outPos = 0;
        while (cursor.remaining() > 0) {
            int tag = cursor.readByte();
            int kind = tag & 0x03;
            if (kind == 0) {
                int len = tag >>> 2;
                if (len < 60) {
                    len += 1;
                } else {
                    int bytes = len - 59;
                    if (bytes > 4 || bytes > cursor.remaining()) {
                        throw new IOException("Corrupt Snappy literal length");
                    }
                    long value = 0;
                    for (int i = 0; i < bytes; i++) {
                        value |= (long) cursor.readByte() << (8 * i);
                    }
                    len = (int) value + 1;
                }
                if (len < 0 || len > cursor.remaining() || len > out.length - outPos) {
                    throw new IOException("Corrupt Snappy literal");
                }
                cursor.copyTo(out, outPos, len);
                outPos += len;
            } else {
                int len;
                int copyOffset;
                if (kind == 1) {
                    len = ((tag >>> 2) & 0x07) + 4;
                    copyOffset = ((tag >>> 5) << 8) | cursor.readByte();
                } else if (kind == 2) {
                    len = (tag >>> 2) + 1;
                    copyOffset = cursor.readByte() | (cursor.readByte() << 8);
                } else {
                    len = (tag >>> 2) + 1;
                    copyOffset = cursor.readByte()
                            | (cursor.readByte() << 8)
                            | (cursor.readByte() << 16)
                            | (cursor.readByte() << 24);
                }
                if (copyOffset <= 0 || copyOffset > outPos || len > out.length - outPos) {
                    throw new IOException("Corrupt Snappy copy");
                }
                int from = outPos - copyOffset;
                for (int i = 0; i < len; i++) {
                    out[outPos++] = out[from + i];
                }
            }
        }
        if (outPos != out.length) {
            throw new IOException("Snappy data ended early: " + outPos + " of " + out.length + " bytes");
        }
        return out;
    }

    private static byte[] gzipDecompress(byte[] data, int offset, int length, int expectedSize)
            throws IOException {
        ByteArrayOutputStream out = expectedSize >= 0
                ? new ByteArrayOutputStream(expectedSize)
                : new ByteArrayOutputStream(Math.max(64, length * 2));
        try (GZIPInputStream in = new GZIPInputStream(new ByteArrayInputStream(data, offset, length))) {
            in.transferTo(out);
        }
        byte[] result = out.toByteArray();
        if (expectedSize >= 0 && result.length != expectedSize) {
            throw new IOException("GZIP page length mismatch: expected " + expectedSize
                    + " but got " + result.length);
        }
        return result;
    }

    private static byte[] lz4RawDecompress(byte[] data, int offset, int length, int expectedSize)
            throws IOException {
        if (expectedSize < 0) {
            throw new IOException("LZ4_RAW pages require a known uncompressed size");
        }
        Cursor cursor = new Cursor(data, offset, offset + length);
        byte[] out = new byte[expectedSize];
        int outPos = 0;
        while (cursor.remaining() > 0) {
            int token = cursor.readByte();
            int literalLength = token >>> 4;
            if (literalLength == 15) {
                int extra;
                do {
                    extra = cursor.readByte();
                    literalLength += extra;
                } while (extra == 255);
            }
            if (literalLength > cursor.remaining() || literalLength > out.length - outPos) {
                throw new IOException("Corrupt LZ4 literal");
            }
            cursor.copyTo(out, outPos, literalLength);
            outPos += literalLength;
            if (cursor.remaining() == 0) {
                break;
            }
            int matchOffset = cursor.readByte() | (cursor.readByte() << 8);
            int matchLength = token & 0x0f;
            if (matchLength == 15) {
                int extra;
                do {
                    extra = cursor.readByte();
                    matchLength += extra;
                } while (extra == 255);
            }
            matchLength += 4;
            if (matchOffset <= 0 || matchOffset > outPos || matchLength > out.length - outPos) {
                throw new IOException("Corrupt LZ4 match");
            }
            int from = outPos - matchOffset;
            for (int i = 0; i < matchLength; i++) {
                out[outPos++] = out[from + i];
            }
        }
        if (outPos != out.length) {
            throw new IOException("LZ4 data ended early: " + outPos + " of " + out.length + " bytes");
        }
        return out;
    }

    private static byte[] slice(byte[] data, int offset, int length) {
        byte[] result = new byte[length];
        System.arraycopy(data, offset, result, 0, length);
        return result;
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

        int remaining() {
            return end - position;
        }

        int readByte() throws IOException {
            if (position >= end) {
                throw new IOException("Unexpected end of compressed page");
            }
            return data[position++] & 0xff;
        }

        void copyTo(byte[] target, int targetOffset, int count) {
            System.arraycopy(data, position, target, targetOffset, count);
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
                    throw new IOException("Snappy varint is too long");
                }
            }
        }
    }
}
