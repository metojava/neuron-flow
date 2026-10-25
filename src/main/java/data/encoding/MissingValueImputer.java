package data.encoding;

/**
 * Missing value imputation for float arrays. Replaces NaN values
 * with the mean or median of the column.
 */
public final class MissingValueImputer {

    private final float[] fillValues;
    private final int features;

    /**
     * Fits the imputer using the mean of each column.
     *
     * @param data [samples][features] with NaN for missing values
     */
    public MissingValueImputer(float[][] data, boolean useMedian) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("data must not be empty");
        }
        this.features = data[0].length;
        this.fillValues = new float[features];

        for (int j = 0; j < features; j++) {
            if (useMedian) {
                float[] col = new float[data.length];
                int count = 0;
                for (int i = 0; i < data.length; i++) {
                    if (!Float.isNaN(data[i][j])) {
                        col[count++] = data[i][j];
                    }
                }
                java.util.Arrays.sort(col, 0, count);
                fillValues[j] = (count == 0) ? 0.0f
                    : (count % 2 == 1)
                        ? col[count / 2]
                        : (col[count / 2 - 1] + col[count / 2]) / 2.0f;
            } else {
                double sum = 0.0;
                int count = 0;
                for (int i = 0; i < data.length; i++) {
                    if (!Float.isNaN(data[i][j])) {
                        sum += data[i][j];
                        count++;
                    }
                }
                fillValues[j] = (count == 0) ? 0.0f : (float) (sum / count);
            }
        }
    }

    /** Mean imputation. */
    public MissingValueImputer(float[][] data) {
        this(data, false);
    }

    /** Imputes missing values in a single sample in place. */
    public void transform(float[] sample) {
        for (int j = 0; j < features; j++) {
            if (Float.isNaN(sample[j])) {
                sample[j] = fillValues[j];
            }
        }
    }

    /** Imputes missing values in a dataset in place. */
    public void transform(float[][] data) {
        for (float[] sample : data) {
            transform(sample);
        }
    }

    public float[] fillValues() {
        return fillValues.clone();
    }
}
