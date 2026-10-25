package layers;

/**
 * Layer normalization: normalizes over all features of a single sample.
 * Uses running or batch statistics. This implementation uses the
 * input's own mean and variance (instance normalization).
 */
public final class LayerNorm {

    private final int features;
    private final float[] gamma;  // scale
    private final float[] beta;   // shift
    private final float epsilon;

    /**
     * @param features number of features
     * @param epsilon  small constant for numerical stability
     */
    public LayerNorm(int features, float epsilon) {
        if (features <= 0) {
            throw new IllegalArgumentException("features must be positive");
        }
        this.features = features;
        this.gamma = new float[features];
        this.beta = new float[features];
        for (int i = 0; i < features; i++) {
            gamma[i] = 1.0f;
        }
        this.epsilon = epsilon;
    }

    /** Default epsilon of 1e-5. */
    public LayerNorm(int features) {
        this(features, 1e-5f);
    }

    /**
     * Replaces the scale and shift parameters, for example with values
     * loaded from a model file.
     *
     * @param gammaValues scale, length features
     * @param betaValues  shift, length features
     * @throws IllegalArgumentException if either length does not match
     */
    public void loadParameters(float[] gammaValues, float[] betaValues) {
        if (gammaValues.length != features || betaValues.length != features) {
            throw new IllegalArgumentException(
                "Expected " + features + " gamma and beta values");
        }
        System.arraycopy(gammaValues, 0, gamma, 0, features);
        System.arraycopy(betaValues, 0, beta, 0, features);
    }

    /**
     * Forward pass: normalizes the input using its own mean and variance.
     *
     * @param input input vector (length features)
     * @return normalized output
     */
    public float[] forward(float[] input) {
        if (input.length != features) {
            throw new IllegalArgumentException(
                "Expected " + features + " features, got " + input.length);
        }
        float mean = 0.0f;
        for (float v : input) mean += v;
        mean /= features;

        float var = 0.0f;
        for (float v : input) {
            float d = v - mean;
            var += d * d;
        }
        var /= features;

        float std = (float) Math.sqrt(var + epsilon);
        float[] out = new float[features];
        for (int i = 0; i < features; i++) {
            out[i] = gamma[i] * (input[i] - mean) / std + beta[i];
        }
        return out;
    }
}
