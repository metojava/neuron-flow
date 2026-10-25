package regularizers;

/**
 * Elastic Net: combination of L1 and L2 penalties.
 * penalty = l1 * sum(|w|) + 0.5 * l2 * sum(w^2).
 * Gives sparsity (L1) with stable correlated-feature handling (L2).
 */
public final class ElasticNetRegularizer {

    private final float l1Lambda;
    private final float l2Lambda;

    public ElasticNetRegularizer(float l1Lambda, float l2Lambda) {
        if (l1Lambda < 0.0f || l2Lambda < 0.0f) {
            throw new IllegalArgumentException(
                "lambdas must be non-negative");
        }
        this.l1Lambda = l1Lambda;
        this.l2Lambda = l2Lambda;
    }

    public static ElasticNetRegularizer of(float l1, float l2) {
        return new ElasticNetRegularizer(l1, l2);
    }

    public float penalty(float[] weights) {
        float sum = 0.0f;
        for (float w : weights) {
            sum += l1Lambda * Math.abs(w)
                 + 0.5f * l2Lambda * w * w;
        }
        return sum;
    }

    public float[] gradient(float[] weights) {
        float[] g = new float[weights.length];
        for (int i = 0; i < weights.length; i++) {
            g[i] = l1Lambda * Math.signum(weights[i])
                 + l2Lambda * weights[i];
        }
        return g;
    }
}
