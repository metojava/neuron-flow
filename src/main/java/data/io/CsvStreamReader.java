package data.io;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Streams a numeric CSV file in batches without loading it fully into memory.
 *
 * <p>Supports a header row (skipped by default), quoted fields with embedded
 * delimiters or newlines, and empty or {@code NA} cells, which become
 * {@link Float#NaN}.
 *
 * <p>Usage:
 * <pre>
 *   try (CsvStreamReader in = new CsvStreamReader(Path.of("big.csv"))) {
 *       float[][] batch;
 *       while ((batch = in.nextBatch(10_000)).length > 0) {
 *           // process batch
 *       }
 *   }
 * </pre>
 */
public final class CsvStreamReader implements Closeable {

    private final BufferedReader reader;
    private final char delimiter;
    private final String[] header;
    private long rowsRead;

    public CsvStreamReader(Path path) throws IOException {
        this(path, ',', true);
    }

    public CsvStreamReader(Path path, char delimiter, boolean hasHeader) throws IOException {
        this.reader = Files.newBufferedReader(path);
        this.delimiter = delimiter;
        this.header = hasHeader ? readFields() : null;
    }

    /** Header names, or {@code null} if the file has no header. */
    public String[] header() {
        return header == null ? null : header.clone();
    }

    /** Number of data rows returned so far. */
    public long rowsRead() {
        return rowsRead;
    }

    /**
     * Reads up to {@code batchSize} rows.
     *
     * @return a matrix with between 0 and batchSize rows; length 0 means end of file
     */
    public float[][] nextBatch(int batchSize) throws IOException {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        List<float[]> rows = new ArrayList<>(Math.min(batchSize, 1024));
        while (rows.size() < batchSize) {
            String[] fields = readFields();
            if (fields == null) break;
            rows.add(parseRow(fields));
        }
        rowsRead += rows.size();
        return rows.toArray(new float[0][]);
    }

    /** Reads the next record, or returns null at end of file. Skips blank lines. */
    private String[] readFields() throws IOException {
        String line;
        do {
            line = reader.readLine();
            if (line == null) return null;
        } while (line.trim().isEmpty());

        // Join physical lines while a quoted field is still open.
        while (countQuotes(line) % 2 != 0) {
            String next = reader.readLine();
            if (next == null) {
                throw new IOException("Unterminated quoted field in CSV");
            }
            line = line + "\n" + next;
        }
        return splitRecord(line);
    }

    private String[] splitRecord(String line) {
        List<String> fields = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        cur.append('"');
                        i++;
                    } else {
                        inQuotes = false;
                    }
                } else {
                    cur.append(c);
                }
            } else if (c == '"') {
                inQuotes = true;
            } else if (c == delimiter) {
                fields.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        fields.add(cur.toString());
        return fields.toArray(new String[0]);
    }

    private float[] parseRow(String[] fields) throws IOException {
        float[] row = new float[fields.length];
        for (int j = 0; j < fields.length; j++) {
            row[j] = parseCell(fields[j]);
        }
        return row;
    }

    static float parseCell(String raw) throws IOException {
        String s = raw.trim();
        if (s.isEmpty() || s.equalsIgnoreCase("NA")) {
            return Float.NaN;
        }
        try {
            return Float.parseFloat(s);
        } catch (NumberFormatException e) {
            throw new IOException("Non-numeric CSV value: '" + raw + "'", e);
        }
    }

    private static int countQuotes(String s) {
        int n = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '"') n++;
        }
        return n;
    }

    @Override
    public void close() throws IOException {
        reader.close();
    }
}
