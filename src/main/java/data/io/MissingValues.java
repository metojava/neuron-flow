package data.io;

import java.util.ArrayList;
import java.util.List;

/**
 * Pandas-style missing value helpers for float matrices. Missing values are
 * represented as {@link Float#NaN}.
 */
public final class MissingValues {

    private MissingValues() {
    }

    /**
     * Returns a copy with every row that contains a NaN removed.
     */
    public static float[][] dropna(float[][] rows) {
        List<float[]> kept = new ArrayList<>(rows.length);
        for (float[] row : rows) {
            if (!hasNaN(row)) kept.add(row);
        }
        return kept.toArray(new float[0][]);
    }

    /**
     * Returns a copy with every NaN replaced by {@code value}.
     */
    public static float[][] fillna(float[][] rows, float value) {
        float[][] out = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            float[] src = rows[i];
            float[] dst = new float[src.length];
            for (int j = 0; j < src.length; j++) {
                dst[j] = Float.isNaN(src[j]) ? value : src[j];
            }
            out[i] = dst;
        }
        return out;
    }

    /**
     * Replaces NaN in each column with that column's mean, computed over the
     * non-NaN values in the given rows.
     */
    public static float[][] fillnaMean(float[][] rows) {
        if (rows.length == 0) return rows;
        int cols = rows[0].length;
        double[] sum = new double[cols];
        int[] count = new int[cols];
        for (float[] row : rows) {
            for (int j = 0; j < cols; j++) {
                if (!Float.isNaN(row[j])) {
                    sum[j] += row[j];
                    count[j]++;
                }
            }
        }
        float[] means = new float[cols];
        for (int j = 0; j < cols; j++) {
            means[j] = count[j] == 0 ? 0f : (float) (sum[j] / count[j]);
        }
        return fillnaColumns(rows, means);
    }

    /**
     * Replaces NaN in column j with {@code values[j]}. Use this to apply
     * statistics computed on a training split to other splits.
     */
    public static float[][] fillnaColumns(float[][] rows, float[] values) {
        float[][] out = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            float[] src = rows[i];
            float[] dst = new float[src.length];
            for (int j = 0; j < src.length; j++) {
                dst[j] = Float.isNaN(src[j]) ? values[j] : src[j];
            }
            out[i] = dst;
        }
        return out;
    }

    private static boolean hasNaN(float[] row) {
        for (float v : row) {
            if (Float.isNaN(v)) return true;
        }
        return false;
    }
}
