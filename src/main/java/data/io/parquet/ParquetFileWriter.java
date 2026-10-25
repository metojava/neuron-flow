package data.io.parquet;

import data.io.parquet.ParquetFormat.ColumnChunk;
import data.io.parquet.ParquetFormat.ColumnMetaData;
import data.io.parquet.ParquetFormat.DataPageHeader;
import data.io.parquet.ParquetFormat.FileMetaData;
import data.io.parquet.ParquetFormat.PageHeader;
import data.io.parquet.ParquetFormat.RowGroup;
import data.io.parquet.ParquetFormat.SchemaElement;
import java.io.BufferedOutputStream;
import java.io.Closeable;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Writes flat, numeric Parquet files using only the JDK.
 *
 * <p>Every column is stored as a nullable {@code FLOAT}. {@link Float#NaN} is
 * written as a Parquet null. The writer emits uncompressed PLAIN data pages
 * with RLE definition levels and groups rows into row groups so that memory use
 * is bounded by {@code rowGroupRows}.
 *
 * <p>This is a filesystem API: no Hadoop classes or configuration are involved.
 */
public final class ParquetFileWriter implements Closeable {

    /** Rows buffered per row group when no explicit size is given. */
    public static final int DEFAULT_ROW_GROUP_ROWS = 64 * 1024;

    private final Path path;
    private final String[] columns;
    private final int rowGroupRows;
    private final OutputStream out;
    private final Counter counter;
    private final List<float[]> pending = new ArrayList<>();
    private final List<RowGroup> rowGroups = new ArrayList<>();
    private final SchemaElement[] schema;
    private long rowsWritten;
    private boolean closed;

    public ParquetFileWriter(Path path, String[] columns, int width) throws IOException {
        this(path, columns, width, DEFAULT_ROW_GROUP_ROWS);
    }

    /**
     * @param path         output file (replaced if it exists)
     * @param columns      column names, or null to use c0, c1, ...
     * @param width        number of columns (ignored when columns is given)
     * @param rowGroupRows rows buffered before a row group is written
     */
    public ParquetFileWriter(Path path, String[] columns, int width, int rowGroupRows)
            throws IOException {
        if (rowGroupRows <= 0) {
            throw new IllegalArgumentException("rowGroupRows must be positive");
        }
        String[] names = columns;
        if (names == null) {
            if (width <= 0) {
                throw new IllegalArgumentException("width must be positive");
            }
            names = new String[width];
            for (int i = 0; i < width; i++) {
                names[i] = "c" + i;
            }
        }
        if (names.length == 0) {
            throw new IllegalArgumentException("At least one column is required");
        }
        this.path = path;
        this.rowGroupRows = rowGroupRows;
        this.columns = names.clone();
        this.schema = new SchemaElement[this.columns.length + 1];
        schema[0] = new SchemaElement(-1, -1, -1, "neuronflow", this.columns.length);
        for (int i = 0; i < this.columns.length; i++) {
            schema[i + 1] = new SchemaElement(ParquetFormat.FLOAT, -1,
                    ParquetFormat.OPTIONAL, sanitize(this.columns[i]), -1);
            this.columns[i] = sanitize(this.columns[i]);
        }

        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        this.out = new BufferedOutputStream(Files.newOutputStream(path), 64 * 1024);
        this.counter = new Counter(out);
        counter.write(ParquetFormat.MAGIC);
    }

    /** Column names as stored in the Parquet schema. */
    public String[] header() {
        return columns.clone();
    }

    public void writeRow(float[] row) throws IOException {
        ensureOpen();
        if (row.length != columns.length) {
            throw new IllegalArgumentException("Expected " + columns.length
                    + " values, got " + row.length);
        }
        pending.add(row.clone());
        rowsWritten++;
        if (pending.size() >= rowGroupRows) {
            flushRowGroup();
        }
    }

    public void writeRows(float[][] rows) throws IOException {
        for (float[] row : rows) {
            writeRow(row);
        }
    }

    public long rowsWritten() {
        return rowsWritten;
    }

    /** Number of row groups already flushed to the file. */
    public int rowGroupsWritten() {
        return rowGroups.size();
    }

    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        try {
            flushRowGroup();
            byte[] footer = ParquetMetadata.serializeFileMetaData(new FileMetaData(
                    1,
                    List.of(schema),
                    rowsWritten,
                    List.copyOf(rowGroups),
                    "neuronflow pure-java-parquet 1.0"));
            counter.write(footer);
            writeIntLE(counter, footer.length);
            counter.write(ParquetFormat.MAGIC);
            counter.flush();
        } finally {
            closed = true;
            out.close();
        }
    }

    private void flushRowGroup() throws IOException {
        if (pending.isEmpty()) {
            return;
        }
        int rows = pending.size();
        List<ColumnChunk> chunks = new ArrayList<>(columns.length);
        long groupBytes = 0;

        for (int c = 0; c < columns.length; c++) {
            byte[] body = buildDataPage(c, rows);
            byte[] pageHeader = ParquetMetadata.serializePageHeader(new PageHeader(
                    ParquetFormat.DATA_PAGE,
                    body.length,
                    body.length,
                    new DataPageHeader(rows, ParquetFormat.PLAIN, ParquetFormat.RLE, ParquetFormat.RLE),
                    null,
                    null));
            long offset = counter.count;
            counter.write(pageHeader);
            counter.write(body);
            long totalSize = (long) pageHeader.length + body.length;
            groupBytes += totalSize;

            ColumnMetaData meta = new ColumnMetaData(
                    ParquetFormat.FLOAT,
                    new int[] {ParquetFormat.PLAIN, ParquetFormat.RLE},
                    new String[] {columns[c]},
                    ParquetFormat.UNCOMPRESSED,
                    rows,
                    totalSize,
                    totalSize,
                    offset,
                    -1);
            chunks.add(new ColumnChunk(meta));
        }

        rowGroups.add(new RowGroup(List.copyOf(chunks), groupBytes, rows));
        pending.clear();
    }

    private byte[] buildDataPage(int column, int rows) throws IOException {
        int[] definitions = new int[rows];
        int values = 0;
        for (int r = 0; r < rows; r++) {
            if (!Float.isNaN(pending.get(r)[column])) {
                definitions[r] = 1;
                values++;
            }
        }
        byte[] levels = Rle.encodeBitPacked(definitions, 1);
        byte[] page = new byte[4 + levels.length + values * 4];
        writeIntLE(page, 0, levels.length);
        System.arraycopy(levels, 0, page, 4, levels.length);
        int pos = 4 + levels.length;
        for (float[] row : pending) {
            float value = row[column];
            if (!Float.isNaN(value)) {
                int bits = Float.floatToRawIntBits(value);
                page[pos++] = (byte) bits;
                page[pos++] = (byte) (bits >>> 8);
                page[pos++] = (byte) (bits >>> 16);
                page[pos++] = (byte) (bits >>> 24);
            }
        }
        return page;
    }

    private void ensureOpen() throws IOException {
        if (closed) {
            throw new IOException("Parquet writer for " + path + " is closed");
        }
    }

    private static void writeIntLE(OutputStream out, int value) throws IOException {
        out.write(value);
        out.write(value >>> 8);
        out.write(value >>> 16);
        out.write(value >>> 24);
    }

    private static void writeIntLE(byte[] target, int offset, int value) {
        target[offset] = (byte) value;
        target[offset + 1] = (byte) (value >>> 8);
        target[offset + 2] = (byte) (value >>> 16);
        target[offset + 3] = (byte) (value >>> 24);
    }

    private static String sanitize(String name) {
        String s = name.replaceAll("[^A-Za-z0-9_]", "_");
        if (s.isEmpty() || Character.isDigit(s.charAt(0))) {
            s = "c_" + s;
        }
        return s;
    }

    private static final class Counter extends FilterOutputStream {
        private long count;

        Counter(OutputStream out) {
            super(out);
        }

        @Override
        public void write(int b) throws IOException {
            out.write(b);
            count++;
        }

        @Override
        public void write(byte[] b, int off, int len) throws IOException {
            out.write(b, off, len);
            count += len;
        }
    }
}
