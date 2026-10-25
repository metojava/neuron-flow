package data.io;

import data.io.parquet.ParquetFileReader;
import data.io.parquet.ParquetFileWriter;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Streaming Parquet support for numeric columns, backed by the pure-Java
 * {@code data.io.parquet} reader and writer. No Hadoop classes are used.
 *
 * <p>Reads INT32, INT64, FLOAT, DOUBLE and BOOLEAN columns as float (booleans
 * become 0/1). Null values become {@link Float#NaN}. Writes every column as
 * nullable FLOAT, so NaN is stored as null.
 *
 * <p>Rows are read one page at a time, so memory use is bounded by one page
 * per column plus the row group being decoded.
 */
public final class ParquetIO {

    private ParquetIO() {
    }

    /** Streams rows from a Parquet file in batches. */
    public static final class Reader implements Closeable {
        private final ParquetFileReader reader;

        public Reader(Path path) throws IOException {
            this.reader = new ParquetFileReader(path);
        }

        /** Column names from the file schema. */
        public String[] header() {
            return reader.header();
        }

        /** Total number of rows in the file. */
        public long numRows() {
            return reader.numRows();
        }

        /**
         * @return up to batchSize rows; length 0 at end of file
         */
        public float[][] nextBatch(int batchSize) throws IOException {
            return reader.nextBatch(batchSize);
        }

        @Override
        public void close() throws IOException {
            reader.close();
        }
    }

    /** Writes numeric rows to a Parquet file. */
    public static final class Writer implements Closeable {
        private final ParquetFileWriter writer;

        /**
         * @param path    output file (replaced if it exists)
         * @param columns column names, or null to use c0, c1, ...
         * @param width   number of columns (ignored when columns is given)
         */
        public Writer(Path path, String[] columns, int width) throws IOException {
            this.writer = new ParquetFileWriter(path, columns, width);
        }

        /**
         * @param path         output file (replaced if it exists)
         * @param columns      column names, or null to use c0, c1, ...
         * @param width        number of columns (ignored when columns is given)
         * @param rowGroupRows rows buffered before a row group is written
         */
        public Writer(Path path, String[] columns, int width, int rowGroupRows) throws IOException {
            this.writer = new ParquetFileWriter(path, columns, width, rowGroupRows);
        }

        public void writeRow(float[] row) throws IOException {
            writer.writeRow(row);
        }

        public void writeRows(float[][] rows) throws IOException {
            writer.writeRows(rows);
        }

        public long rowsWritten() {
            return writer.rowsWritten();
        }

        @Override
        public void close() throws IOException {
            writer.close();
        }
    }
}
