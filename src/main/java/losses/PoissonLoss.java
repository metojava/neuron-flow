package losses;

/**
 * Poisson negative log-likelihood: predicts = mean(pred - target * log(pred)).
 * Used for count data (event frequency prediction). Predictions
 * must be positive (use a softplus or exp output activation).
 */
public final class PoissonLoss implements LossFunction {

    public static PoissonLoss of() {
        return new PoissonLoss();
    }

    @Override
    public float loss(float[] p, float[] t) {
        checkShapes(p, t);
        float sum = 0.0f;
        float eps = 1e-7f;

        for (int i = 0; i < p.length; i++) {
            float pi = Math.max(p[i], eps);
            sum += pi - t[i] * (float) Math.log(pi);
        }

        return sum / p.length;
    }

    @Override
    public float[] gradient(float[] p, float[] t) {
        checkShapes(p, t);
        float[] g = new float[p.length];
        float eps = 1e-7f;

        for (int i = 0; i < p.length; i++) {
            float pi = Math.max(p[i], eps);
            g[i] = (1.0f - t[i] / pi) / p.length;
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
