package regularizers;

/**
 * L1 (Lasso) penalty: lambda * sum(|w|). Encourages sparsity —
 * many weights are driven exactly to zero. Add the returned
 * penalty to the loss and the returned gradient to the weight
 * gradient during training.
 */
public final class L1Regularizer {

    private final float lambda;

    public L1Regularizer(float lambda) {
        if (lambda < 0.0f) {
            throw new IllegalArgumentException(
                "lambda must be non-negative");
        }
        this.lambda = lambda;
    }

    public static L1Regularizer of(float lambda) {
        return new L1Regularizer(lambda);
    }

    /**
     * Penalty value: lambda * sum(|w|).
     *
     * @param weights all trainable weights (biases usually
     *                excluded)
     * @return scalar penalty
     */
    public float penalty(float[] weights) {
        float sum = 0.0f;
        for (float w : weights) {
            sum += Math.abs(w);
        }
        return lambda * sum;
    }

    /**
     * Gradient of the penalty w.r.t. each weight:
     * lambda * sign(w).
     *
     * @param weights all trainable weights
     * @return one gradient value per weight
     */
    public float[] gradient(float[] weights) {
        float[] g = new float[weights.length];
        for (int i = 0; i < weights.length; i++) {
            g[i] = lambda * Math.signum(weights[i]);
        }
        return g;
    }
}
