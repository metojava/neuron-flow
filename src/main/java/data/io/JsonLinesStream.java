package data.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Streaming JSON Lines (.jsonl) support. Each line is one row, written as a
 * flat array of numbers, for example {@code [0.1, 0.2, null]}. {@code null}
 * is read and written as {@link Float#NaN}.
 *
 * <p>Only flat numeric arrays are supported. Objects and strings are rejected.
 */
public final class JsonLinesStream {

    private JsonLinesStream() {
    }

    /** Streams rows from a .jsonl file in batches. */
    public static final class Reader implements Closeable {
        private final BufferedReader reader;
        private long lineNo;

        public Reader(Path path) throws IOException {
            this.reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);
        }

        /**
         * @return up to batchSize rows; length 0 at end of file
         */
        public float[][] nextBatch(int batchSize) throws IOException {
            if (batchSize <= 0) {
                throw new IllegalArgumentException("batchSize must be positive");
            }
            List<float[]> rows = new ArrayList<>(Math.min(batchSize, 1024));
            String line;
            while (rows.size() < batchSize && (line = reader.readLine()) != null) {
                lineNo++;
                if (line.trim().isEmpty()) continue;
                rows.add(parseLine(line, lineNo));
            }
            return rows.toArray(new float[0][]);
        }

        @Override
        public void close() throws IOException {
            reader.close();
        }
    }

    /** Streams rows to a .jsonl file. */
    public static final class Writer implements Closeable {
        private final BufferedWriter writer;
        private long rowsWritten;

        public Writer(Path path) throws IOException {
            this.writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8);
        }

        public void writeRow(float[] row) throws IOException {
            StringBuilder sb = new StringBuilder(row.length * 8 + 2);
            sb.append('[');
            for (int j = 0; j < row.length; j++) {
                if (j > 0) sb.append(',');
                if (Float.isNaN(row[j]) || Float.isInfinite(row[j])) {
                    sb.append("null");
                } else {
                    sb.append(row[j]);
                }
            }
            sb.append(']');
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

        @Override
        public void close() throws IOException {
            writer.close();
        }
    }

    static float[] parseLine(String line, long lineNo) throws IOException {
        String s = line.trim();
        if (s.length() < 2 || s.charAt(0) != '[' || s.charAt(s.length() - 1) != ']') {
            throw new IOException("Line " + lineNo + ": expected a JSON array");
        }
        String body = s.substring(1, s.length() - 1).trim();
        if (body.isEmpty()) {
            return new float[0];
        }
        String[] parts = body.split(",", -1);
        float[] row = new float[parts.length];
        for (int j = 0; j < parts.length; j++) {
            String p = parts[j].trim();
            if (p.equals("null")) {
                row[j] = Float.NaN;
                continue;
            }
            try {
                row[j] = Float.parseFloat(p);
            } catch (NumberFormatException e) {
                throw new IOException("Line " + lineNo + ": non-numeric value '" + p + "'", e);
            }
        }
        return row;
    }
}
