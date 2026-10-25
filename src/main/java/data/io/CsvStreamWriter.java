package data.io;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Streams numeric rows to a CSV file. Rows can be written one at a time or
 * in batches, so the full dataset never has to be in memory.
 *
 * <p>{@link Float#NaN} is written as an empty cell, which the reader maps back
 * to NaN.
 */
public final class CsvStreamWriter implements Closeable {

    private final BufferedWriter writer;
    private final char delimiter;
    private long rowsWritten;

    public CsvStreamWriter(Path path) throws IOException {
        this(path, ',', null, false);
    }

    /**
     * @param path      output file
     * @param delimiter column separator
     * @param header    column names to write first, or null for none
     * @param append    true to append to an existing file
     */
    public CsvStreamWriter(Path path, char delimiter, String[] header, boolean append)
            throws IOException {
        this.delimiter = delimiter;
        this.writer = append
                ? Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND)
                : Files.newBufferedWriter(path, StandardCharsets.UTF_8);
        if (header != null) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < header.length; i++) {
                if (i > 0) sb.append(delimiter);
                sb.append(escape(header[i]));
            }
            writer.write(sb.toString());
            writer.newLine();
        }
    }

    public void writeRow(float[] row) throws IOException {
        StringBuilder sb = new StringBuilder(row.length * 8);
        for (int j = 0; j < row.length; j++) {
            if (j > 0) sb.append(delimiter);
            if (!Float.isNaN(row[j])) {
                sb.append(row[j]);
            }
        }
        writer.write(sb.toString());
        writer.newLine();
        rowsWritten++;
    }

    public void writeRows(float[][] rows) throws IOException {
        for (float[] row : rows) {
            writeRow(row);
        }
    }

    public long rowsWritten() {
        return rowsWritten;
    }

    private String escape(String s) {
        boolean needsQuotes = s.indexOf(delimiter) >= 0 || s.indexOf('"') >= 0
                || s.indexOf('\n') >= 0 || s.indexOf('\r') >= 0;
        if (!needsQuotes) return s;
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    @Override
    public void close() throws IOException {
        writer.close();
    }
}
