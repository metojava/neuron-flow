package data.io.parquet;

import data.io.parquet.ParquetFormat.ColumnChunk;
import data.io.parquet.ParquetFormat.ColumnMetaData;
import data.io.parquet.ParquetFormat.DataPageHeader;
import data.io.parquet.ParquetFormat.DataPageHeaderV2;
import data.io.parquet.ParquetFormat.DictionaryPageHeader;
import data.io.parquet.ParquetFormat.FileMetaData;
import data.io.parquet.ParquetFormat.PageHeader;
import data.io.parquet.ParquetFormat.RowGroup;
import data.io.parquet.ParquetFormat.SchemaElement;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/** Reads and writes the Parquet thrift metadata without Hadoop. */
final class ParquetMetadata {

    private ParquetMetadata() {
    }

    static byte[] serializeFileMetaData(FileMetaData metadata) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        writeFileMetaData(metadata, bytes);
        return bytes.toByteArray();
    }

    static FileMetaData readFileMetaData(byte[] bytes) throws IOException {
        return readFileMetaData(new ByteArrayInputStream(bytes));
    }

    static void writeFileMetaData(FileMetaData metadata, OutputStream out) throws IOException {
        ThriftCompact.Writer w = new ThriftCompact.Writer(out);
        w.structBegin();
        w.i32Field(1, metadata.version());
        w.listField(2, ThriftCompact.STRUCT, metadata.schema().size());
        for (SchemaElement element : metadata.schema()) {
            writeSchemaElement(w, element);
        }
        w.i64Field(3, metadata.numRows());
        w.listField(4, ThriftCompact.STRUCT, metadata.rowGroups().size());
        for (RowGroup group : metadata.rowGroups()) {
            writeRowGroup(w, group);
        }
        if (metadata.createdBy() != null) {
            w.stringField(6, metadata.createdBy());
        }
        w.structEnd();
    }

    static byte[] serializePageHeader(PageHeader header) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        writePageHeader(header, bytes);
        return bytes.toByteArray();
    }

    static void writePageHeader(PageHeader header, OutputStream out) throws IOException {
        ThriftCompact.Writer w = new ThriftCompact.Writer(out);
        w.structBegin();
        w.i32Field(1, header.type());
        w.i32Field(2, header.uncompressedPageSize());
        w.i32Field(3, header.compressedPageSize());
        if (header.dataPageHeader() != null) {
            w.structField(5);
            w.structBegin();
            DataPageHeader page = header.dataPageHeader();
            w.i32Field(1, page.numValues());
            w.i32Field(2, page.encoding());
            w.i32Field(3, page.definitionLevelEncoding());
            w.i32Field(4, page.repetitionLevelEncoding());
            w.structEnd();
        }
        if (header.dictionaryPageHeader() != null) {
            w.structField(7);
            w.structBegin();
            DictionaryPageHeader page = header.dictionaryPageHeader();
            w.i32Field(1, page.numValues());
            w.i32Field(2, page.encoding());
            w.boolField(3, page.isSorted());
            w.structEnd();
        }
        if (header.dataPageHeaderV2() != null) {
            w.structField(8);
            w.structBegin();
            DataPageHeaderV2 page = header.dataPageHeaderV2();
            w.i32Field(1, page.numValues());
            w.i32Field(2, page.numNulls());
            w.i32Field(3, page.numRows());
            w.i32Field(4, page.encoding());
            w.i32Field(5, page.definitionLevelsByteLength());
            w.i32Field(6, page.repetitionLevelsByteLength());
            w.boolField(7, page.isCompressed());
            w.structEnd();
        }
        w.structEnd();
    }

    static PageHeader readPageHeader(InputStream in) throws IOException {
        ThriftCompact.Reader r = new ThriftCompact.Reader(in);
        return readPageHeader(r);
    }

    static PageHeader readPageHeader(ThriftCompact.Reader r) throws IOException {
        int type = -1;
        int uncompressedPageSize = 0;
        int compressedPageSize = 0;
        DataPageHeader dataPage = null;
        DictionaryPageHeader dictionaryPage = null;
        DataPageHeaderV2 dataPageV2 = null;

        r.structBegin();
        int fieldType;
        while ((fieldType = r.fieldBegin()) != ThriftCompact.STOP) {
            switch (r.fieldId()) {
                case 1 -> type = r.readI32();
                case 2 -> uncompressedPageSize = r.readI32();
                case 3 -> compressedPageSize = r.readI32();
                case 5 -> dataPage = readDataPageHeader(r);
                case 7 -> dictionaryPage = readDictionaryPageHeader(r);
                case 8 -> dataPageV2 = readDataPageHeaderV2(r);
                default -> r.skip(fieldType);
            }
        }
        r.structEnd();
        return new PageHeader(type, uncompressedPageSize, compressedPageSize,
                dataPage, dictionaryPage, dataPageV2);
    }

    private static FileMetaData readFileMetaData(InputStream in) throws IOException {
        ThriftCompact.Reader r = new ThriftCompact.Reader(in);
        int version = 0;
        List<SchemaElement> schema = new ArrayList<>();
        long numRows = 0;
        List<RowGroup> rowGroups = new ArrayList<>();
        String createdBy = null;

        r.structBegin();
        int fieldType;
        while ((fieldType = r.fieldBegin()) != ThriftCompact.STOP) {
            switch (r.fieldId()) {
                case 1 -> version = r.readI32();
                case 2 -> {
                    ThriftCompact.ListHeader list = r.listBegin();
                    schema = new ArrayList<>(list.size());
                    for (int i = 0; i < list.size(); i++) {
                        schema.add(readSchemaElement(r));
                    }
                }
                case 3 -> numRows = r.readI64();
                case 4 -> {
                    ThriftCompact.ListHeader list = r.listBegin();
                    rowGroups = new ArrayList<>(list.size());
                    for (int i = 0; i < list.size(); i++) {
                        rowGroups.add(readRowGroup(r));
                    }
                }
                case 6 -> createdBy = r.readString();
                default -> r.skip(fieldType);
            }
        }
        r.structEnd();
        return new FileMetaData(version, schema, numRows, rowGroups, createdBy);
    }

    private static void writeSchemaElement(ThriftCompact.Writer w, SchemaElement element)
            throws IOException {
        w.structBegin();
        if (element.type() >= 0) {
            w.i32Field(1, element.type());
        }
        if (element.typeLength() >= 0) {
            w.i32Field(2, element.typeLength());
        }
        if (element.repetition() >= 0) {
            w.i32Field(3, element.repetition());
        }
        w.stringField(4, element.name());
        if (element.numChildren() >= 0) {
            w.i32Field(5, element.numChildren());
        }
        w.structEnd();
    }

    private static SchemaElement readSchemaElement(ThriftCompact.Reader r) throws IOException {
        int type = -1;
        int typeLength = -1;
        int repetition = -1;
        String name = "";
        int numChildren = -1;

        r.structBegin();
        int fieldType;
        while ((fieldType = r.fieldBegin()) != ThriftCompact.STOP) {
            switch (r.fieldId()) {
                case 1 -> type = r.readI32();
                case 2 -> typeLength = r.readI32();
                case 3 -> repetition = r.readI32();
                case 4 -> name = r.readString();
                case 5 -> numChildren = r.readI32();
                default -> r.skip(fieldType);
            }
        }
        r.structEnd();
        return new SchemaElement(type, typeLength, repetition, name, numChildren);
    }

    private static void writeRowGroup(ThriftCompact.Writer w, RowGroup group) throws IOException {
        w.structBegin();
        w.listField(1, ThriftCompact.STRUCT, group.columns().size());
        for (ColumnChunk column : group.columns()) {
            writeColumnChunk(w, column);
        }
        w.i64Field(2, group.totalByteSize());
        w.i64Field(3, group.numRows());
        w.structEnd();
    }

    private static RowGroup readRowGroup(ThriftCompact.Reader r) throws IOException {
        List<ColumnChunk> columns = new ArrayList<>();
        long totalByteSize = 0;
        long numRows = 0;

        r.structBegin();
        int fieldType;
        while ((fieldType = r.fieldBegin()) != ThriftCompact.STOP) {
            switch (r.fieldId()) {
                case 1 -> {
                    ThriftCompact.ListHeader list = r.listBegin();
                    columns = new ArrayList<>(list.size());
                    for (int i = 0; i < list.size(); i++) {
                        columns.add(readColumnChunk(r));
                    }
                }
                case 2 -> totalByteSize = r.readI64();
                case 3 -> numRows = r.readI64();
                default -> r.skip(fieldType);
            }
        }
        r.structEnd();
        return new RowGroup(columns, totalByteSize, numRows);
    }

    private static void writeColumnChunk(ThriftCompact.Writer w, ColumnChunk chunk)
            throws IOException {
        w.structBegin();
        w.i64Field(2, 0);
        if (chunk.metaData() != null) {
            w.structField(3);
            w.structBegin();
            ColumnMetaData meta = chunk.metaData();
            w.i32Field(1, meta.type());
            w.listField(2, ThriftCompact.I32, meta.encodings().length);
            for (int encoding : meta.encodings()) {
                w.writeI32(encoding);
            }
            w.listField(3, ThriftCompact.BINARY, meta.path().length);
            for (String path : meta.path()) {
                w.writeString(path);
            }
            w.i32Field(4, meta.codec());
            w.i64Field(5, meta.numValues());
            w.i64Field(6, meta.totalUncompressedSize());
            w.i64Field(7, meta.totalCompressedSize());
            w.i64Field(9, meta.dataPageOffset());
            if (meta.dictionaryPageOffset() >= 0) {
                w.i64Field(11, meta.dictionaryPageOffset());
            }
            w.structEnd();
        }
        w.structEnd();
    }

    private static ColumnChunk readColumnChunk(ThriftCompact.Reader r) throws IOException {
        ColumnMetaData meta = null;
        r.structBegin();
        int fieldType;
        while ((fieldType = r.fieldBegin()) != ThriftCompact.STOP) {
            if (r.fieldId() == 3) {
                meta = readColumnMetaData(r);
            } else {
                r.skip(fieldType);
            }
        }
        r.structEnd();
        return new ColumnChunk(meta);
    }

    private static ColumnMetaData readColumnMetaData(ThriftCompact.Reader r) throws IOException {
        int type = -1;
        int[] encodings = new int[0];
        String[] path = new String[0];
        int codec = ParquetFormat.UNCOMPRESSED;
        long numValues = 0;
        long totalUncompressedSize = 0;
        long totalCompressedSize = 0;
        long dataPageOffset = -1;
        long dictionaryPageOffset = -1;

        r.structBegin();
        int fieldType;
        while ((fieldType = r.fieldBegin()) != ThriftCompact.STOP) {
            switch (r.fieldId()) {
                case 1 -> type = r.readI32();
                case 2 -> {
                    ThriftCompact.ListHeader list = r.listBegin();
                    encodings = new int[list.size()];
                    for (int i = 0; i < list.size(); i++) {
                        encodings[i] = r.readI32();
                    }
                }
                case 3 -> {
                    ThriftCompact.ListHeader list = r.listBegin();
                    path = new String[list.size()];
                    for (int i = 0; i < list.size(); i++) {
                        path[i] = r.readString();
                    }
                }
                case 4 -> codec = r.readI32();
                case 5 -> numValues = r.readI64();
                case 6 -> totalUncompressedSize = r.readI64();
                case 7 -> totalCompressedSize = r.readI64();
                case 9 -> dataPageOffset = r.readI64();
                case 11 -> dictionaryPageOffset = r.readI64();
                default -> r.skip(fieldType);
            }
        }
        r.structEnd();
        return new ColumnMetaData(type, encodings, path, codec, numValues,
                totalUncompressedSize, totalCompressedSize, dataPageOffset, dictionaryPageOffset);
    }

    private static DataPageHeader readDataPageHeader(ThriftCompact.Reader r) throws IOException {
        int numValues = 0;
        int encoding = ParquetFormat.PLAIN;
        int definitionLevelEncoding = ParquetFormat.RLE;
        int repetitionLevelEncoding = ParquetFormat.RLE;

        r.structBegin();
        int fieldType;
        while ((fieldType = r.fieldBegin()) != ThriftCompact.STOP) {
            switch (r.fieldId()) {
                case 1 -> numValues = r.readI32();
                case 2 -> encoding = r.readI32();
                case 3 -> definitionLevelEncoding = r.readI32();
                case 4 -> repetitionLevelEncoding = r.readI32();
                default -> r.skip(fieldType);
            }
        }
        r.structEnd();
        return new DataPageHeader(numValues, encoding, definitionLevelEncoding, repetitionLevelEncoding);
    }

    private static DictionaryPageHeader readDictionaryPageHeader(ThriftCompact.Reader r)
            throws IOException {
        int numValues = 0;
        int encoding = ParquetFormat.PLAIN;
        boolean isSorted = false;

        r.structBegin();
        int fieldType;
        while ((fieldType = r.fieldBegin()) != ThriftCompact.STOP) {
            switch (r.fieldId()) {
                case 1 -> numValues = r.readI32();
                case 2 -> encoding = r.readI32();
                case 3 -> isSorted = r.readBoolField(fieldType);
                default -> r.skip(fieldType);
            }
        }
        r.structEnd();
        return new DictionaryPageHeader(numValues, encoding, isSorted);
    }

    private static DataPageHeaderV2 readDataPageHeaderV2(ThriftCompact.Reader r) throws IOException {
        int numValues = 0;
        int numNulls = 0;
        int numRows = 0;
        int encoding = ParquetFormat.PLAIN;
        int definitionLevelsByteLength = 0;
        int repetitionLevelsByteLength = 0;
        boolean isCompressed = true;

        r.structBegin();
        int fieldType;
        while ((fieldType = r.fieldBegin()) != ThriftCompact.STOP) {
            switch (r.fieldId()) {
                case 1 -> numValues = r.readI32();
                case 2 -> numNulls = r.readI32();
                case 3 -> numRows = r.readI32();
                case 4 -> encoding = r.readI32();
                case 5 -> definitionLevelsByteLength = r.readI32();
                case 6 -> repetitionLevelsByteLength = r.readI32();
                case 7 -> isCompressed = r.readBoolField(fieldType);
                default -> r.skip(fieldType);
            }
        }
        r.structEnd();
        return new DataPageHeaderV2(numValues, numNulls, numRows, encoding,
                definitionLevelsByteLength, repetitionLevelsByteLength, isCompressed);
    }
}
