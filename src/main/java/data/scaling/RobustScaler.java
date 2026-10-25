package data.scaling;

import java.util.Arrays;

/**
 * Robust scaler: scales features using median and interquartile range,
 * so outliers have less influence than with MinMax or StandardScaler.
 * Fit on training data only, then transform both train and test.
 */
public final class RobustScaler {

    private final float[] median;
    private final float[] iqr;
    private final int features;

    /**
     * Fits the scaler on the given data.
     *
     * @param data [samples][features]
     */
    public RobustScaler(float[][] data) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("data must not be empty");
        }
        this.features = data[0].length;
        this.median = new float[features];
        this.iqr = new float[features];

        for (int j = 0; j < features; j++) {
            float[] col = new float[data.length];
            for (int i = 0; i < data.length; i++) {
                col[i] = data[i][j];
            }
            Arrays.sort(col);
            median[j] = percentile(col, 50);
            float q1 = percentile(col, 25);
            float q3 = percentile(col, 75);
            iqr[j] = q3 - q1;
            if (iqr[j] == 0.0f) {
                iqr[j] = 1.0f;
            }
        }
    }

    private static float percentile(float[] sorted, int p) {
        int n = sorted.length;
        double pos = (p / 100.0) * (n - 1);
        int lo = (int) Math.floor(pos);
        int hi = (int) Math.ceil(pos);
        if (lo == hi) return sorted[lo];
        double frac = pos - lo;
        return (float) (sorted[lo] * (1 - frac) + sorted[hi] * frac);
    }

    /** Transforms a single sample in place. */
    public void transform(float[] sample) {
        for (int j = 0; j < features; j++) {
            sample[j] = (sample[j] - median[j]) / iqr[j];
        }
    }

    /** Transforms a dataset in place. */
    public void transform(float[][] data) {
        for (float[] sample : data) {
            transform(sample);
        }
    }

    public float[] median() {
        return median.clone();
    }

    public float[] iqr() {
        return iqr.clone();
    }
}
