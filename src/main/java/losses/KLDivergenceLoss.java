package losses;

/**
 * Kullback-Leibler divergence between two probability
 * distributions: KL(t || p) = sum t[i] * log(t[i] / p[i]).
 * Predictions and targets are treated as probability vectors
 * (should sum to 1). Common in variational autoencoders and
 * distillation.
 */
public final class KLDivergenceLoss implements LossFunction {

    public static KLDivergenceLoss of() {
        return new KLDivergenceLoss();
    }

    @Override
    public float loss(float[] p, float[] t) {
        checkShapes(p, t);
        float sum = 0.0f;
        float eps = 1e-7f;

        for (int i = 0; i < p.length; i++) {
            float pi = Math.max(p[i], eps);
            float ti = Math.max(t[i], eps);
            sum += ti * (float) Math.log(ti / pi);
        }

        return sum;
    }

    @Override
    public float[] gradient(float[] p, float[] t) {
        checkShapes(p, t);
        float[] g = new float[p.length];
        float eps = 1e-7f;

        for (int i = 0; i < p.length; i++) {
            float pi = Math.max(p[i], eps);
            g[i] = -t[i] / pi;
        }

        return g;
    }

    private static void checkShapes(float[] p, float[] t) {
        if (p == null || t == null || p.length == 0
                || p.length != t.length) {
            throw new IllegalArgumentException(
                "Predictions and targets must have equal nonzero lengths");
        }
    }
}
