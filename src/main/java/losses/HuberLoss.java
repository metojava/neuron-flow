package losses;

/**
 * Huber loss (Smooth L1): quadratic for small errors, linear for
 * large ones. Combines the best of MSE (smooth near zero) and
 * MAE (robust to outliers). Delta controls the transition point.
 * Defaults to delta = 1.0 (PyTorch SmoothL1Loss).
 */
public final class HuberLoss implements LossFunction {

    private final float delta;

    public HuberLoss(float delta) {
        if (delta <= 0.0f) {
            throw new IllegalArgumentException(
                "delta must be positive");
        }
        this.delta = delta;
    }

    public static HuberLoss of() {
        return new HuberLoss(1.0f);
    }

    public static HuberLoss of(float delta) {
        return new HuberLoss(delta);
    }

    @Override
    public float loss(float[] p, float[] t) {
        checkShapes(p, t);
        float sum = 0.0f;

        for (int i = 0; i < p.length; i++) {
            float d = p[i] - t[i];
            float ad = Math.abs(d);

            if (ad <= delta) {
                sum += 0.5f * d * d / delta;
            } else {
                sum += ad - 0.5f * delta;
            }
        }

        return sum / p.length;
    }

    @Override
    public float[] gradient(float[] p, float[] t) {
        checkShapes(p, t);
        float[] g = new float[p.length];

        for (int i = 0; i < p.length; i++) {
            float d = p[i] - t[i];
            float ad = Math.abs(d);

            if (ad <= delta) {
                g[i] = d / delta;
            } else {
                g[i] = Math.signum(d);
            }
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
