package data.loaders;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Minimal JSON array loader. Expects a flat array of numbers or an
 * array of arrays of numbers. Returns float[][].
 *
 * <p>Example file:
 * <pre>
 * [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
 * </pre>
 *
 * <p>This is a lightweight parser; it does not handle objects, strings,
 * or nested arrays deeper than two levels.
 */
public final class JsonArrayLoader {

    private JsonArrayLoader() {
    }

    /**
     * Loads a JSON file containing a 2-D array of numbers.
     *
     * @param path path to the JSON file
     * @return [rows][cols] float matrix
     * @throws IOException if the file cannot be read
     */
    public static float[][] load2D(Path path) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
        }
        String json = sb.toString().trim();
        if (!json.startsWith("[") || !json.endsWith("]")) {
            throw new IllegalArgumentException("Expected a JSON array");
        }
        // Remove outer brackets
        json = json.substring(1, json.length() - 1).trim();
        if (json.isEmpty()) {
            return new float[0][];
        }

        List<float[]> rows = new ArrayList<>();
        // Parse row by row: each row is [a, b, c]
        int depth = 0;
        int start = -1;
        for (int i = 0; i < json.length(); i++) {
            char ch = json.charAt(i);
            if (ch == '[') {
                if (depth == 0) start = i;
                depth++;
            } else if (ch == ']') {
                depth--;
                if (depth == 0 && start >= 0) {
                    String rowStr = json.substring(start + 1, i);
                    rows.add(parseRow(rowStr));
                    start = -1;
                }
            }
        }
        // If no nested arrays, treat as flat 1-D array
        if (rows.isEmpty()) {
            rows.add(parseRow(json));
        }
        return rows.toArray(new float[0][]);
    }

    private static float[] parseRow(String rowStr) {
        String[] parts = rowStr.split(",");
        List<Float> values = new ArrayList<>();
        for (String part : parts) {
            part = part.trim();
            if (part.isEmpty()) continue;
            values.add(Float.parseFloat(part));
        }
        float[] result = new float[values.size()];
        for (int i = 0; i < values.size(); i++) {
            result[i] = values.get(i);
        }
        return result;
    }
}
