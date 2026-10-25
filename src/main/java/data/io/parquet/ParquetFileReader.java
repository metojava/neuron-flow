package data.io.parquet;

import data.io.parquet.ParquetFormat.ColumnMetaData;
import data.io.parquet.ParquetFormat.DataPageHeader;
import data.io.parquet.ParquetFormat.DataPageHeaderV2;
import data.io.parquet.ParquetFormat.DictionaryPageHeader;
import data.io.parquet.ParquetFormat.FileMetaData;
import data.io.parquet.ParquetFormat.PageHeader;
import data.io.parquet.ParquetFormat.RowGroup;
import data.io.parquet.ParquetFormat.SchemaElement;
import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads flat, numeric Parquet files using only the JDK.
 *
 * <p>Columns of physical type INT32, INT64, FLOAT, DOUBLE and BOOLEAN are
 * exposed as {@code float} (booleans become 0/1). Nulls become
 * {@link Float#NaN}. Pages are decoded lazily, so at most one page per column
 * is held in memory at a time.
 *
 * <p>Supported encodings are PLAIN, PLAIN_DICTIONARY/RLE_DICTIONARY and RLE
 * (booleans); supported codecs are UNCOMPRESSED, SNAPPY, GZIP and LZ4_RAW.
 * This is a filesystem API: no Hadoop classes or configuration are involved.
 */
public final class ParquetFileReader implements Closeable {

    private final Path path;
    private final FileChannel channel;
    private final FileMetaData metadata;
    private final String[] columns;
    private final int[] types;
    private final int[] maxDefinitionLevels;
    private int rowGroupIndex = -1;
    private long rowsRemaining;
    private ColumnReader[] columnReaders;
    private boolean closed;

    public ParquetFileReader(Path path) throws IOException {
        this.path = path;
        this.channel = FileChannel.open(path, StandardOpenOption.READ);
        try {
            long size = channel.size();
            if (size < 12) {
                throw new IOException("Not a Parquet file: " + path);
            }
            byte[] head = readAt(channel, 0, 4);
            requireMagic(head, 0, path);
            byte[] tail = readAt(channel, size - 8, 8);
            requireMagic(tail, 4, path);
            int footerLength = readIntLE(tail, 0);
            if (footerLength < 0 || footerLength > size - 12) {
                throw new IOException("Invalid Parquet footer length " + footerLength + " in " + path);
            }
            byte[] footer = readAt(channel, size - 8 - footerLength, footerLength);
            this.metadata = ParquetMetadata.readFileMetaData(footer);
            validateSchema(metadata.schema());
            int count = metadata.schema().size() - 1;
            // Filter out unsupported column types
            List<String> supportedColumns = new ArrayList<>();
            List<Integer> supportedTypes = new ArrayList<>();
            List<Integer> supportedMaxDefLevels = new ArrayList<>();
            for (int i = 0; i < count; i++) {
                SchemaElement element = metadata.schema().get(i + 1);
                int type = element.type();
                if (isSupportedType(type)) {
                    supportedColumns.add(element.name());
                    supportedTypes.add(type);
                    supportedMaxDefLevels.add(element.repetition() == ParquetFormat.OPTIONAL ? 1 : 0);
                } else {
                    System.err.println("Skipping unsupported Parquet column type "
                            + ParquetFormat.typeName(type) + " for column " + element.name());
                }
            }
            this.columns = supportedColumns.toArray(new String[0]);
            this.types = supportedTypes.stream().mapToInt(i -> i).toArray();
            this.maxDefinitionLevels = supportedMaxDefLevels.stream().mapToInt(i -> i).toArray();
        } catch (IOException | RuntimeException e) {
            channel.close();
            throw e;
        }
    }

    private static boolean isSupportedType(int type) {
        return type == ParquetFormat.BOOLEAN || type == ParquetFormat.INT32
                || type == ParquetFormat.INT64 || type == ParquetFormat.FLOAT
                || type == ParquetFormat.DOUBLE;
    }

    /** Column names from the file schema. */
    public String[] header() {
        return columns.clone();
    }

    public long numRows() {
        return metadata.numRows();
    }

    public int rowGroupCount() {
        return metadata.rowGroups().size();
    }

    /**
     * @return up to batchSize rows; length 0 at end of file
     */
    public float[][] nextBatch(int batchSize) throws IOException {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        ensureOpen();
        List<float[]> rows = new ArrayList<>(Math.min(batchSize, 4096));
        while (rows.size() < batchSize) {
            if (rowsRemaining == 0) {
                if (!openNextRowGroup()) {
                    break;
                }
            }
            int take = (int) Math.min(batchSize - rows.size(), rowsRemaining);
            for (int r = 0; r < take; r++) {
                float[] row = new float[columns.length];
                for (int c = 0; c < columns.length; c++) {
                    row[c] = columnReaders[c].nextValue();
                }
                rows.add(row);
            }
            rowsRemaining -= take;
        }
        return rows.toArray(new float[0][]);
    }

    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        channel.close();
    }

    private boolean openNextRowGroup() throws IOException {
        List<RowGroup> groups = metadata.rowGroups();
        while (++rowGroupIndex < groups.size()) {
            RowGroup group = groups.get(rowGroupIndex);
            if (group.numRows() <= 0) {
                continue;
            }
            if (group.columns().size() < columns.length) {
                throw new IOException("Row group " + rowGroupIndex + " has "
                        + group.columns().size() + " columns, expected at least " + columns.length);
            }
            columnReaders = new ColumnReader[columns.length];
            for (int c = 0; c < columns.length; c++) {
                ColumnMetaData meta = group.columns().get(c).metaData();
                if (meta == null) {
                    throw new IOException("Missing column metadata in row group " + rowGroupIndex);
                }
                columnReaders[c] = new ColumnReader(meta, types[c], maxDefinitionLevels[c], columns[c]);
            }
            rowsRemaining = group.numRows();
            return true;
        }
        return false;
    }

    private void ensureOpen() throws IOException {
        if (closed) {
            throw new IOException("Parquet reader for " + path + " is closed");
        }
    }

    private static void validateSchema(List<SchemaElement> schema) throws IOException {
        if (schema.isEmpty()) {
            throw new IOException("Parquet file has an empty schema");
        }
        SchemaElement root = schema.get(0);
        int children = root.numChildren() >= 0 ? root.numChildren() : schema.size() - 1;
        if (schema.size() != children + 1) {
            throw new IOException("Nested Parquet schemas are not supported by this reader");
        }
        for (int i = 1; i < schema.size(); i++) {
            SchemaElement element = schema.get(i);
            if (element.numChildren() > 0) {
                throw new IOException("Nested Parquet schemas are not supported by this reader");
            }
            if (element.repetition() == ParquetFormat.REPEATED) {
                throw new IOException("Repeated Parquet columns are not supported: " + element.name());
            }
            switch (element.type()) {
                case ParquetFormat.BOOLEAN, ParquetFormat.INT32, ParquetFormat.INT64,
                        ParquetFormat.FLOAT, ParquetFormat.DOUBLE -> {
                }
                default -> {
                    // Skip unsupported column types (e.g., BYTE_ARRAY for strings)
                    System.err.println("Skipping unsupported Parquet column type "
                            + ParquetFormat.typeName(element.type()) + " for column " + element.name());
                }
            }
        }
    }

    private static void requireMagic(byte[] bytes, int offset, Path path) throws IOException {
        for (int i = 0; i < 4; i++) {
            if (bytes[offset + i] != ParquetFormat.MAGIC[i]) {
                throw new IOException("Not a Parquet file: " + path);
            }
        }
    }

    private static byte[] readAt(FileChannel channel, long position, int length) throws IOException {
        if (length < 0) {
            throw new IOException("Invalid read length " + length);
        }
        ByteBuffer buffer = ByteBuffer.allocate(length);
        long positionInFile = position;
        while (buffer.hasRemaining()) {
            int n = channel.read(buffer, positionInFile);
            if (n < 0) {
                throw new EOFException("Unexpected end of Parquet file");
            }
            positionInFile += n;
        }
        return buffer.array();
    }

    private static int readIntLE(byte[] data, int offset) {
        return (data[offset] & 0xff)
                | ((data[offset + 1] & 0xff) << 8)
                | ((data[offset + 2] & 0xff) << 16)
                | ((data[offset + 3] & 0xff) << 24);
    }

    private static long readLongLE(byte[] data, int offset) {
        return (readIntLE(data, offset) & 0xffffffffL)
                | ((long) readIntLE(data, offset + 4) << 32);
    }

    private final class ColumnReader {
        private final ColumnMetaData meta;
        private final int type;
        private final int maxDefinitionLevel;
        private final String name;
        private long streamPosition;
        private float[] values = new float[0];
        private int valueIndex;
        private long decodedValues;
        private float[] dictionary;

        ColumnReader(ColumnMetaData meta, int type, int maxDefinitionLevel, String name)
                throws IOException {
            this.meta = meta;
            this.type = type;
            this.maxDefinitionLevel = maxDefinitionLevel;
            this.name = name;
            long dataOffset = meta.dataPageOffset();
            long dictionaryOffset = meta.dictionaryPageOffset();
            long start = dataOffset;
            if (dictionaryOffset >= 0 && (start < 0 || dictionaryOffset < start)) {
                start = dictionaryOffset;
            }
            if (start < 0) {
                throw new IOException("Column " + name + " has no page offset");
            }
            this.streamPosition = start;
        }

        float nextValue() throws IOException {
            while (valueIndex >= values.length) {
                if (decodedValues >= meta.numValues()) {
                    throw new IOException("Column " + name + " ended before its row group");
                }
                readNextDataPage();
            }
            return values[valueIndex++];
        }

        private void readNextDataPage() throws IOException {
            while (true) {
                SeekableBufferedInput input = new SeekableBufferedInput(channel, streamPosition);
                PageHeader header = ParquetMetadata.readPageHeader(input);
                long headerLength = input.consumed();
                if (headerLength <= 0) {
                    throw new IOException("Invalid page header in column " + name);
                }
                streamPosition += headerLength;
                if (header.compressedPageSize() < 0 || header.uncompressedPageSize() < 0) {
                    throw new IOException("Invalid page size in column " + name);
                }
                byte[] page = readAt(channel, streamPosition, header.compressedPageSize());
                streamPosition += page.length;

                switch (header.type()) {
                    case ParquetFormat.DICTIONARY_PAGE -> {
                        DictionaryPageHeader dictionaryHeader = header.dictionaryPageHeader();
                        if (dictionaryHeader == null) {
                            throw new IOException("Missing dictionary page header in column " + name);
                        }
                        byte[] decoded = ParquetCompression.decompress(meta.codec(), page, 0,
                                page.length, header.uncompressedPageSize());
                        dictionary = decodePlain(decoded, 0, decoded.length,
                                dictionaryHeader.numValues(), type);
                    }
                    case ParquetFormat.DATA_PAGE -> {
                        DataPageHeader dataHeader = header.dataPageHeader();
                        if (dataHeader == null) {
                            throw new IOException("Missing data page header in column " + name);
                        }
                        byte[] decoded = ParquetCompression.decompress(meta.codec(), page, 0,
                                page.length, header.uncompressedPageSize());
                        values = decodeDataPageV1(decoded, dataHeader);
                        valueIndex = 0;
                        decodedValues += dataHeader.numValues();
                        if (decodedValues > meta.numValues()) {
                            throw new IOException("Column " + name + " contains more values than its metadata");
                        }
                        return;
                    }
                    case ParquetFormat.DATA_PAGE_V2 -> {
                        DataPageHeaderV2 dataHeader = header.dataPageHeaderV2();
                        if (dataHeader == null) {
                            throw new IOException("Missing data page v2 header in column " + name);
                        }
                        values = decodeDataPageV2(page, dataHeader, header.uncompressedPageSize());
                        valueIndex = 0;
                        decodedValues += dataHeader.numValues();
                        if (decodedValues > meta.numValues()) {
                            throw new IOException("Column " + name + " contains more values than its metadata");
                        }
                        return;
                    }
                    case ParquetFormat.INDEX_PAGE -> {
                        // Page indexes are not needed for sequential row access.
                    }
                    default -> throw new IOException("Unsupported Parquet page type "
                            + header.type() + " in column " + name);
                }
            }
        }

        private float[] decodeDataPageV1(byte[] page, DataPageHeader header) throws IOException {
            int numValues = header.numValues();
            if (numValues < 0) {
                throw new IOException("Invalid data page value count in column " + name);
            }
            int offset = 0;
            if (maxDefinitionLevel > 0) {
                if (page.length < 4) {
                    throw new IOException("Truncated definition levels in column " + name);
                }
                int length = readIntLE(page, 0);
                offset = 4;
                if (length < 0 || length > page.length - offset) {
                    throw new IOException("Invalid definition level length in column " + name);
                }
                int[] definitions = Rle.decode(page, offset, length,
                        bitsNeeded(maxDefinitionLevel), numValues);
                offset += length;
                return assembleValues(page, offset, page.length - offset,
                        header.encoding(), definitions);
            }
            int[] definitions = new int[numValues];
            return assembleValues(page, offset, page.length - offset,
                    header.encoding(), definitions);
        }

        private float[] decodeDataPageV2(byte[] page, DataPageHeaderV2 header,
                                         int uncompressedPageSize) throws IOException {
            int numValues = header.numValues();
            int repetitionLength = header.repetitionLevelsByteLength();
            int definitionLength = header.definitionLevelsByteLength();
            if (numValues < 0 || repetitionLength < 0 || definitionLength < 0
                    || (long) repetitionLength + definitionLength > page.length) {
                throw new IOException("Invalid data page v2 lengths in column " + name);
            }
            int[] definitions;
            if (maxDefinitionLevel > 0) {
                definitions = Rle.decode(page, repetitionLength, definitionLength,
                        bitsNeeded(maxDefinitionLevel), numValues);
            } else {
                definitions = new int[numValues];
            }
            int levelBytes = repetitionLength + definitionLength;
            int valueBytes = page.length - levelBytes;
            byte[] data;
            if (header.isCompressed() && meta.codec() != ParquetFormat.UNCOMPRESSED) {
                int expected = uncompressedPageSize - levelBytes;
                data = ParquetCompression.decompress(meta.codec(), page, levelBytes,
                        valueBytes, expected);
            } else {
                data = slice(page, levelBytes, valueBytes);
            }
            return assembleValues(data, 0, data.length, header.encoding(), definitions);
        }

        private float[] assembleValues(byte[] data, int offset, int length,
                                       int encoding, int[] definitions) throws IOException {
            int present = 0;
            for (int definition : definitions) {
                if (definition >= maxDefinitionLevel) {
                    present++;
                }
            }
            float[] decoded = decodeValues(data, offset, length, encoding, present);
            float[] result = new float[definitions.length];
            int next = 0;
            for (int i = 0; i < definitions.length; i++) {
                result[i] = definitions[i] >= maxDefinitionLevel ? decoded[next++] : Float.NaN;
            }
            return result;
        }

        private float[] decodeValues(byte[] data, int offset, int length,
                                     int encoding, int count) throws IOException {
            if (count == 0) {
                return new float[0];
            }
            switch (encoding) {
                case ParquetFormat.PLAIN -> {
                    return decodePlain(data, offset, length, count, type);
                }
                case ParquetFormat.PLAIN_DICTIONARY, ParquetFormat.RLE_DICTIONARY -> {
                    if (dictionary == null) {
                        throw new IOException("Dictionary-encoded page without dictionary in column " + name);
                    }
                    if (length < 1) {
                        throw new IOException("Truncated dictionary indices in column " + name);
                    }
                    int bitWidth = data[offset] & 0xff;
                    int[] indices = Rle.decode(data, offset + 1, length - 1, bitWidth, count);
                    float[] result = new float[count];
                    for (int i = 0; i < count; i++) {
                        int index = indices[i];
                        if (index < 0 || index >= dictionary.length) {
                            throw new IOException("Dictionary index " + index
                                    + " out of range in column " + name);
                        }
                        result[i] = dictionary[index];
                    }
                    return result;
                }
                case ParquetFormat.RLE -> {
                    if (type != ParquetFormat.BOOLEAN) {
                        throw new IOException("RLE value encoding is only supported for BOOLEAN in column " + name);
                    }
                    if (length < 4) {
                        throw new IOException("Truncated RLE boolean data in column " + name);
                    }
                    int runLength = readIntLE(data, offset);
                    if (runLength < 0 || runLength > length - 4) {
                        throw new IOException("Invalid RLE boolean length in column " + name);
                    }
                    int[] bits = Rle.decode(data, offset + 4, runLength, 1, count);
                    float[] result = new float[count];
                    for (int i = 0; i < count; i++) {
                        result[i] = bits[i] != 0 ? 1f : 0f;
                    }
                    return result;
                }
                default -> throw new IOException("Unsupported Parquet value encoding "
                        + encoding + " in column " + name);
            }
        }

        private float[] decodePlain(byte[] data, int offset, int length,
                                    int count, int physicalType) throws IOException {
            int width = switch (physicalType) {
                case ParquetFormat.BOOLEAN -> 0;
                case ParquetFormat.INT32, ParquetFormat.FLOAT -> 4;
                case ParquetFormat.INT64, ParquetFormat.DOUBLE -> 8;
                default -> -1;
            };
            if (width < 0) {
                throw new IOException("Unsupported PLAIN type "
                        + ParquetFormat.typeName(physicalType) + " in column " + name);
            }
            if (width == 0) {
                int required = (count + 7) / 8;
                if (length < required) {
                    throw new IOException("Truncated boolean data in column " + name);
                }
                float[] result = new float[count];
                for (int i = 0; i < count; i++) {
                    result[i] = ((data[offset + (i >>> 3)] >>> (i & 7)) & 1) != 0 ? 1f : 0f;
                }
                return result;
            }
            if ((long) count * width > length) {
                throw new IOException("Truncated PLAIN data in column " + name);
            }
            float[] result = new float[count];
            for (int i = 0; i < count; i++) {
                int at = offset + i * width;
                result[i] = switch (physicalType) {
                    case ParquetFormat.FLOAT -> Float.intBitsToFloat(readIntLE(data, at));
                    case ParquetFormat.DOUBLE -> (float) Double.longBitsToDouble(readLongLE(data, at));
                    case ParquetFormat.INT32 -> readIntLE(data, at);
                    case ParquetFormat.INT64 -> (float) readLongLE(data, at);
                    default -> throw new IOException("Unsupported PLAIN type");
                };
            }
            return result;
        }
    }

    private static int bitsNeeded(int value) {
        return value == 0 ? 0 : 32 - Integer.numberOfLeadingZeros(value);
    }

    private static byte[] slice(byte[] data, int offset, int length) {
        byte[] result = new byte[length];
        System.arraycopy(data, offset, result, 0, length);
        return result;
    }

    /**
     * Reads exactly the bytes the thrift parser asks for while tracking how
     * many bytes were consumed, so that the page payload starts at the right
     * file offset.
     */
    private static final class SeekableBufferedInput extends InputStream {
        private static final int BUFFER_SIZE = 8192;

        private final FileChannel channel;
        private final long base;
        private final byte[] buffer = new byte[BUFFER_SIZE];
        private long logicalPosition;
        private int bufferPosition;
        private int bufferLength;

        SeekableBufferedInput(FileChannel channel, long base) {
            this.channel = channel;
            this.base = base;
        }

        long consumed() {
            return logicalPosition;
        }

        @Override
        public int read() throws IOException {
            if (bufferPosition >= bufferLength) {
                if (!fill()) {
                    return -1;
                }
            }
            long consumed = logicalPosition;
            logicalPosition++;
            return buffer[bufferPosition++] & 0xff;
        }

        @Override
        public int read(byte[] target, int off, int len) throws IOException {
            if (len == 0) {
                return 0;
            }
            int total = 0;
            while (total < len) {
                if (bufferPosition >= bufferLength && !fill()) {
                    break;
                }
                int n = Math.min(len - total, bufferLength - bufferPosition);
                System.arraycopy(buffer, bufferPosition, target, off + total, n);
                bufferPosition += n;
                logicalPosition += n;
                total += n;
            }
            return total == 0 ? -1 : total;
        }

        private boolean fill() throws IOException {
            int limit = 0;
            long position = base + logicalPosition;
            while (limit < buffer.length) {
                int n = channel.read(ByteBuffer.wrap(buffer, limit, buffer.length - limit), position);
                if (n < 0) {
                    break;
                }
                if (n == 0) {
                    break;
                }
                limit += n;
                position += n;
            }
            bufferPosition = 0;
            bufferLength = limit;
            return limit > 0;
        }
    }
}
