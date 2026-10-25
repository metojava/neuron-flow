package data.io.parquet;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Minimal implementation of the Thrift compact protocol, which Parquet uses
 * for its file metadata and page headers.
 */
final class ThriftCompact {

    static final int STOP = 0;
    static final int BOOLEAN_TRUE = 1;
    static final int BOOLEAN_FALSE = 2;
    static final int BYTE = 3;
    static final int I16 = 4;
    static final int I32 = 5;
    static final int I64 = 6;
    static final int DOUBLE = 7;
    static final int BINARY = 8;
    static final int LIST = 9;
    static final int SET = 10;
    static final int MAP = 11;
    static final int STRUCT = 12;

    private ThriftCompact() {
    }

    static final class Writer {
        private final OutputStream out;
        private final Deque<Integer> lastFieldId = new ArrayDeque<>();

        Writer(OutputStream out) {
            this.out = out;
        }

        void structBegin() {
            lastFieldId.push(0);
        }

        void structEnd() throws IOException {
            out.write(STOP);
            lastFieldId.pop();
        }

        void fieldBegin(int id, int type) throws IOException {
            int last = lastFieldId.pop();
            int delta = id - last;
            if (delta > 0 && delta <= 15) {
                out.write((delta << 4) | type);
            } else {
                out.write(type);
                writeZigZag(id);
            }
            lastFieldId.push(id);
        }

        void boolField(int id, boolean value) throws IOException {
            fieldBegin(id, value ? BOOLEAN_TRUE : BOOLEAN_FALSE);
        }

        void i32Field(int id, int value) throws IOException {
            fieldBegin(id, I32);
            writeZigZag(value);
        }

        void i64Field(int id, long value) throws IOException {
            fieldBegin(id, I64);
            writeZigZag64(value);
        }

        void i16Field(int id, int value) throws IOException {
            fieldBegin(id, I16);
            writeZigZag(value);
        }

        void stringField(int id, String value) throws IOException {
            fieldBegin(id, BINARY);
            writeBinary(value.getBytes(StandardCharsets.UTF_8));
        }

        void structField(int id) throws IOException {
            fieldBegin(id, STRUCT);
        }

        void listField(int id, int elementType, int size) throws IOException {
            fieldBegin(id, LIST);
            listBegin(elementType, size);
        }

        void listBegin(int elementType, int size) throws IOException {
            if (size <= 14) {
                out.write((size << 4) | elementType);
            } else {
                out.write(0xf0 | elementType);
                writeVarint(size);
            }
        }

        void writeI32(int value) throws IOException {
            writeZigZag(value);
        }

        void writeI64(long value) throws IOException {
            writeZigZag64(value);
        }

        void writeBinary(byte[] value) throws IOException {
            writeVarint(value.length);
            out.write(value);
        }

        void writeString(String value) throws IOException {
            writeBinary(value.getBytes(StandardCharsets.UTF_8));
        }

        private void writeZigZag(int value) throws IOException {
            writeVarint(((value << 1) ^ (value >> 31)) & 0xffffffffL);
        }

        private void writeZigZag64(long value) throws IOException {
            writeVarint((value << 1) ^ (value >> 63));
        }

        private void writeVarint(long value) throws IOException {
            while ((value & ~0x7fL) != 0) {
                out.write((int) ((value & 0x7f) | 0x80));
                value >>>= 7;
            }
            out.write((int) value);
        }
    }

    record ListHeader(int size, int elementType) {
    }

    static final class Reader {
        private final InputStream in;
        private final Deque<Integer> lastFieldId = new ArrayDeque<>();
        private int currentType;
        private int currentFieldId;

        Reader(InputStream in) {
            this.in = in;
        }

        void structBegin() {
            lastFieldId.push(0);
        }

        void structEnd() {
            lastFieldId.pop();
        }

        /** @return the field type, or {@link #STOP} at the end of a struct */
        int fieldBegin() throws IOException {
            int header = in.read();
            if (header < 0) {
                throw new EOFException("Unexpected end of thrift data");
            }
            if (header == STOP) {
                currentType = STOP;
                return STOP;
            }
            int type = header & 0x0f;
            int delta = (header >>> 4) & 0x0f;
            int last = lastFieldId.pop();
            int id = delta == 0 ? readI16Value() : last + delta;
            lastFieldId.push(id);
            currentType = type;
            currentFieldId = id;
            return type;
        }

        int currentType() {
            return currentType;
        }

        int fieldId() {
            return currentFieldId;
        }

        boolean readBoolField(int type) {
            return type == BOOLEAN_TRUE;
        }

        int readI16() throws IOException {
            return readI16Value();
        }

        int readI32() throws IOException {
            return (int) readZigZag();
        }

        long readI64() throws IOException {
            return readZigZag();
        }

        String readString() throws IOException {
            return new String(readBinary(), StandardCharsets.UTF_8);
        }

        byte[] readBinary() throws IOException {
            long length = readVarint();
            if (length < 0 || length > Integer.MAX_VALUE) {
                throw new IOException("Invalid thrift binary length " + length);
            }
            byte[] data = new byte[(int) length];
            readFully(data, 0, data.length);
            return data;
        }

        ListHeader listBegin() throws IOException {
            int header = in.read();
            if (header < 0) {
                throw new EOFException("Unexpected end of thrift list");
            }
            int size = (header >>> 4) & 0x0f;
            int type = header & 0x0f;
            if (size == 15) {
                long full = readVarint();
                if (full < 0 || full > Integer.MAX_VALUE) {
                    throw new IOException("Invalid thrift list size " + full);
                }
                size = (int) full;
            }
            return new ListHeader(size, type);
        }

        void skip(int type) throws IOException {
            switch (type) {
                case STOP, BOOLEAN_TRUE, BOOLEAN_FALSE -> {
                    // Booleans carry their value in the field header.
                }
                case BYTE -> in.read();
                case I16, I32, I64 -> readVarint();
                case DOUBLE -> skipFully(8);
                case BINARY -> {
                    long length = readVarint();
                    if (length < 0) {
                        throw new IOException("Invalid thrift binary length " + length);
                    }
                    skipFully(length);
                }
                case LIST, SET -> {
                    ListHeader list = listBegin();
                    for (int i = 0; i < list.size(); i++) {
                        skipElement(list.elementType);
                    }
                }
                case MAP -> {
                    long size = readVarint();
                    if (size > 0) {
                        int types = in.read();
                        if (types < 0) {
                            throw new EOFException("Unexpected end of thrift map");
                        }
                        int keyType = (types >>> 4) & 0x0f;
                        int valueType = types & 0x0f;
                        for (long i = 0; i < size; i++) {
                            skipElement(keyType);
                            skipElement(valueType);
                        }
                    }
                }
                case STRUCT -> {
                    structBegin();
                    int fieldType;
                    while ((fieldType = fieldBegin()) != STOP) {
                        skip(fieldType);
                    }
                    structEnd();
                }
                default -> throw new IOException("Unknown thrift type " + type);
            }
        }

        private void skipElement(int type) throws IOException {
            if (type == BOOLEAN_TRUE || type == BOOLEAN_FALSE) {
                int value = in.read();
                if (value < 0) {
                    throw new EOFException("Unexpected end of thrift boolean");
                }
            } else {
                skip(type);
            }
        }

        private int readI16Value() throws IOException {
            return (int) readZigZag();
        }

        private long readZigZag() throws IOException {
            long raw = readVarint();
            return (raw >>> 1) ^ -(raw & 1);
        }

        private long readVarint() throws IOException {
            long result = 0;
            int shift = 0;
            while (true) {
                int b = in.read();
                if (b < 0) {
                    throw new EOFException("Unexpected end of thrift varint");
                }
                result |= (long) (b & 0x7f) << shift;
                if ((b & 0x80) == 0) {
                    return result;
                }
                shift += 7;
                if (shift > 63) {
                    throw new IOException("Thrift varint is too long");
                }
            }
        }

        private void readFully(byte[] data, int off, int len) throws IOException {
            while (len > 0) {
                int n = in.read(data, off, len);
                if (n < 0) {
                    throw new EOFException("Unexpected end of thrift data");
                }
                off += n;
                len -= n;
            }
        }

        private void skipFully(long len) throws IOException {
            while (len > 0) {
                long skipped = in.skip(len);
                if (skipped <= 0) {
                    if (in.read() < 0) {
                        throw new EOFException("Unexpected end of thrift data");
                    }
                    skipped = 1;
                }
                len -= skipped;
            }
        }
    }
}
