package data.scaling;

/**
 * Min-max scaler: scales features to [0, 1] using training-set min and max.
 * Fit on training data only, then transform both train and test.
 */
public final class MinMaxScaler {

    private final float[] min;
    private final float[] max;
    private final int features;

    /**
     * Fits the scaler on the given data.
     *
     * @param data [samples][features]
     * @throws IllegalArgumentException on null or empty data
     */
    public MinMaxScaler(float[][] data) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("data must not be empty");
        }
        this.features = data[0].length;
        this.min = new float[features];
        this.max = new float[features];
        System.arraycopy(data[0], 0, min, 0, features);
        System.arraycopy(data[0], 0, max, 0, features);

        for (int i = 1; i < data.length; i++) {
            for (int j = 0; j < features; j++) {
                if (data[i][j] < min[j]) min[j] = data[i][j];
                if (data[i][j] > max[j]) max[j] = data[i][j];
            }
        }
    }

    /** Transforms a single sample in place. */
    public void transform(float[] sample) {
        for (int j = 0; j < features; j++) {
            float range = max[j] - min[j];
            if (range == 0.0f) {
                sample[j] = 0.0f;
            } else {
                sample[j] = (sample[j] - min[j]) / range;
            }
        }
    }

    /** Transforms a dataset in place. */
    public void transform(float[][] data) {
        for (float[] sample : data) {
            transform(sample);
        }
    }

    /** Returns the fitted minimum per feature. */
    public float[] min() {
        return min.clone();
    }

    /** Returns the fitted maximum per feature. */
    public float[] max() {
        return max.clone();
    }
}
