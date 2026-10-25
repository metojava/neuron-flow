package regularizers;

/**
 * L2 (Ridge / weight decay) penalty: 0.5 * lambda * sum(w^2).
 * Shrinks weights toward zero without driving them exactly to
 * zero. The most common regularizer for dense networks.
 */
public final class L2Regularizer {

    private final float lambda;

    public L2Regularizer(float lambda) {
        if (lambda < 0.0f) {
            throw new IllegalArgumentException(
                "lambda must be non-negative");
        }
        this.lambda = lambda;
    }

    public static L2Regularizer of(float lambda) {
        return new L2Regularizer(lambda);
    }

    /**
     * Penalty value: 0.5 * lambda * sum(w^2).
     *
     * @param weights all trainable weights
     * @return scalar penalty
     */
    public float penalty(float[] weights) {
        float sum = 0.0f;
        for (float w : weights) {
            sum += w * w;
        }
        return 0.5f * lambda * sum;
    }

    /**
     * Gradient of the penalty w.r.t. each weight: lambda * w.
     *
     * @param weights all trainable weights
     * @return one gradient value per weight
     */
    public float[] gradient(float[] weights) {
        float[] g = new float[weights.length];
        for (int i = 0; i < weights.length; i++) {
            g[i] = lambda * weights[i];
        }
        return g;
    }
}
