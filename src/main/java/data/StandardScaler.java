package data;

/**
 * Standardizes each feature using statistics fitted on training data.
 * Constant features receive a scale of one and therefore become zero.
 */
public final class StandardScaler {
    private final double[] means;
    private final double[] scales;

    private StandardScaler(double[] means, double[] scales) {
        this.means = means;
        this.scales = scales;
    }

    /** Fits feature statistics without modifying the supplied rows. */
    public static StandardScaler fit(float[][] rows) {
        int width = validate(rows);
        double[] means = new double[width];
        double[] squared = new double[width];
        int count = 0;
        for (float[] row : rows) {
            count++;
            for (int j = 0; j < width; j++) {
                double delta = row[j] - means[j];
                means[j] += delta / count;
                squared[j] += delta * (row[j] - means[j]);
            }
        }
        double[] scales = new double[width];
        for (int j = 0; j < width; j++) {
            double std = Math.sqrt(squared[j] / count);
            scales[j] = std == 0.0 ? 1.0 : std;
        }
        return new StandardScaler(means, scales);
    }

    /** Returns scaled copies of the rows; use fitted training statistics for test data. */
    public float[][] transform(float[][] rows) {
        checkWidth(rows);
        float[][] result = new float[rows.length][means.length];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < means.length; j++) {
                result[i][j] = (float) ((rows[i][j] - means[j]) / scales[j]);
            }
        }
        return result;
    }

    /** Reverses a previous transformation, returning new rows. */
    public float[][] inverseTransform(float[][] rows) {
        checkWidth(rows);
        float[][] result = new float[rows.length][means.length];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < means.length; j++) {
                result[i][j] = (float) (rows[i][j] * scales[j] + means[j]);
            }
        }
        return result;
    }

    public int featureCount() {
        return means.length;
    }

    private void checkWidth(float[][] rows) {
        if (validate(rows) != means.length) {
            throw new IllegalArgumentException("Feature count differs from fitted data");
        }
    }

    private static int validate(float[][] rows) {
        if (rows == null || rows.length == 0 || rows[0] == null
            || rows[0].length == 0) {
            throw new IllegalArgumentException("Data must be nonempty");
        }
        int width = rows[0].length;
        for (float[] row : rows) {
            if (row == null || row.length != width) {
                throw new IllegalArgumentException("Rows must have equal feature counts");
            }
            for (float value : row) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("Features must be finite");
                }
            }
        }
        return width;
    }
}
