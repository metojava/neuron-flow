package data.io.parquet;

import java.util.List;

/**
 * Constants and metadata structures from the Parquet file format.
 *
 * <p>Only the subset needed by the numeric column API is modelled. Unknown
 * thrift fields are skipped while reading.
 */
final class ParquetFormat {

    static final byte[] MAGIC = {'P', 'A', 'R', '1'};

    // Physical types (Type enum)
    static final int BOOLEAN = 0;
    static final int INT32 = 1;
    static final int INT64 = 2;
    static final int INT96 = 3;
    static final int FLOAT = 4;
    static final int DOUBLE = 5;
    static final int BYTE_ARRAY = 6;
    static final int FIXED_LEN_BYTE_ARRAY = 7;

    // FieldRepetitionType enum
    static final int REQUIRED = 0;
    static final int OPTIONAL = 1;
    static final int REPEATED = 2;

    // Encoding enum
    static final int PLAIN = 0;
    static final int PLAIN_DICTIONARY = 2;
    static final int RLE = 3;
    static final int BIT_PACKED = 4;
    static final int RLE_DICTIONARY = 8;

    // CompressionCodec enum
    static final int UNCOMPRESSED = 0;
    static final int SNAPPY = 1;
    static final int GZIP = 2;
    static final int LZO = 3;
    static final int BROTLI = 4;
    static final int LZ4 = 5;
    static final int ZSTD = 6;
    static final int LZ4_RAW = 7;

    // PageType enum
    static final int DATA_PAGE = 0;
    static final int INDEX_PAGE = 1;
    static final int DICTIONARY_PAGE = 2;
    static final int DATA_PAGE_V2 = 3;

    private ParquetFormat() {
    }

    record SchemaElement(int type, int typeLength, int repetition, String name, int numChildren) {
    }

    record ColumnMetaData(int type,
                          int[] encodings,
                          String[] path,
                          int codec,
                          long numValues,
                          long totalUncompressedSize,
                          long totalCompressedSize,
                          long dataPageOffset,
                          long dictionaryPageOffset) {
    }

    record ColumnChunk(ColumnMetaData metaData) {
    }

    record RowGroup(List<ColumnChunk> columns, long totalByteSize, long numRows) {
    }

    record FileMetaData(int version,
                        List<SchemaElement> schema,
                        long numRows,
                        List<RowGroup> rowGroups,
                        String createdBy) {
    }

    record DataPageHeader(int numValues,
                          int encoding,
                          int definitionLevelEncoding,
                          int repetitionLevelEncoding) {
    }

    record DictionaryPageHeader(int numValues, int encoding, boolean isSorted) {
    }

    record DataPageHeaderV2(int numValues,
                            int numNulls,
                            int numRows,
                            int encoding,
                            int definitionLevelsByteLength,
                            int repetitionLevelsByteLength,
                            boolean isCompressed) {
    }

    record PageHeader(int type,
                      int uncompressedPageSize,
                      int compressedPageSize,
                      DataPageHeader dataPageHeader,
                      DictionaryPageHeader dictionaryPageHeader,
                      DataPageHeaderV2 dataPageHeaderV2) {
    }

    static String codecName(int codec) {
        return switch (codec) {
            case UNCOMPRESSED -> "UNCOMPRESSED";
            case SNAPPY -> "SNAPPY";
            case GZIP -> "GZIP";
            case LZO -> "LZO";
            case BROTLI -> "BROTLI";
            case LZ4 -> "LZ4";
            case ZSTD -> "ZSTD";
            case LZ4_RAW -> "LZ4_RAW";
            default -> "codec " + codec;
        };
    }

    static String typeName(int type) {
        return switch (type) {
            case BOOLEAN -> "BOOLEAN";
            case INT32 -> "INT32";
            case INT64 -> "INT64";
            case INT96 -> "INT96";
            case FLOAT -> "FLOAT";
            case DOUBLE -> "DOUBLE";
            case BYTE_ARRAY -> "BYTE_ARRAY";
            case FIXED_LEN_BYTE_ARRAY -> "FIXED_LEN_BYTE_ARRAY";
            default -> "type " + type;
        };
    }
}
