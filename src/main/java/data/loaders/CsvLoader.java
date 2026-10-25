package data.loaders;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Loads a CSV file into float arrays. The first row is treated as a
 * header and skipped. Each subsequent row must have the same number
 * of columns; non-numeric values cause an exception.
 *
 * <p>Usage:
 * <pre>
 *   CsvLoader loader = new CsvLoader(Path.of("data.csv"), ',');
 *   float[][] features = loader.columns(0, 3);  // columns 0..2
 *   float[]   labels   = loader.column(4);      // column 4
 * </pre>
 */
public final class CsvLoader {

    private final String[][] rows;
    private final String[] header;

    /**
     * @param path      path to the CSV file
     * @param delimiter column separator
     * @throws IOException if the file cannot be read
     */
    public CsvLoader(Path path, char delimiter) throws IOException {
        List<String[]> parsed = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                parsed.add(line.split(String.valueOf(delimiter), -1));
            }
        }
        if (parsed.isEmpty()) {
            throw new IllegalArgumentException("CSV file is empty");
        }
        this.header = parsed.get(0);
        this.rows = parsed.subList(1, parsed.size()).toArray(new String[0][]);
    }

    /** Returns the header row. */
    public String[] header() {
        return header.clone();
    }

    /** Number of data rows. */
    public int rowCount() {
        return rows.length;
    }

    /** Number of columns. */
    public int columnCount() {
        return header.length;
    }

    /**
     * Returns a single column as a float array.
     *
     * @param index zero-based column index
     * @return float array of length rowCount()
     */
    public float[] column(int index) {
        float[] result = new float[rows.length];
        for (int i = 0; i < rows.length; i++) {
            result[i] = Float.parseFloat(rows[i][index].trim());
        }
        return result;
    }

    /**
     * Returns a range of columns as a float matrix.
     *
     * @param from first column (inclusive)
     * @param to   last column (exclusive)
     * @return [rowCount()][to-from] matrix
     */
    public float[][] columns(int from, int to) {
        float[][] result = new float[rows.length][to - from];
        for (int i = 0; i < rows.length; i++) {
            for (int j = from; j < to; j++) {
                result[i][j - from] = Float.parseFloat(rows[i][j].trim());
            }
        }
        return result;
    }

    /**
     * Returns all columns as a float matrix.
     *
     * @return [rowCount()][columnCount()] matrix
     */
    public float[][] all() {
        return columns(0, header.length);
    }
}
