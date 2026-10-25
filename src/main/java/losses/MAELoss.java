package losses;

/**
 * Mean Absolute Error (L1 loss).
 */
public final class MAELoss implements LossFunction {

    public static MAELoss of() {
        return new MAELoss();
    }

    @Override
    public float loss(float[] p, float[] t) {
        checkShapes(p, t);
        float sum = 0.0f;

        for (int i = 0; i < p.length; i++) {
            sum += Math.abs(p[i] - t[i]);
        }

        return sum / p.length;
    }

    @Override
    public float[] gradient(float[] p, float[] t) {
        checkShapes(p, t);
        float[] g = new float[p.length];

        for (int i = 0; i < p.length; i++) {
            float d = p[i] - t[i];
            if (d > 0.0f) {
                g[i] = 1.0f / p.length;
            } else if (d < 0.0f) {
                g[i] = -1.0f / p.length;
            } else {
                g[i] = 0.0f;
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
