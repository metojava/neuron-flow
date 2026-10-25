package losses;

/**
 * Multi-class hinge loss: max(0, 1 - score[correct] + max_{j != correct} score[j]).
 * Commonly used with SVM-style classifiers. Targets are one-hot;
 * the gradient is sparse (only fires when the margin is violated).
 */
public final class HingeLoss implements LossFunction {

    public static HingeLoss of() {
        return new HingeLoss();
    }

    @Override
    public float loss(float[] p, float[] t) {
        checkShapes(p, t);

        int correct = argmax(t);
        float maxOther = Float.NEGATIVE_INFINITY;

        for (int i = 0; i < p.length; i++) {
            if (i != correct && p[i] > maxOther) {
                maxOther = p[i];
            }
        }

        float margin = 1.0f - p[correct] + maxOther;
        return Math.max(0.0f, margin);
    }

    @Override
    public float[] gradient(float[] p, float[] t) {
        checkShapes(p, t);
        float[] g = new float[p.length];

        int correct = argmax(t);
        int bestOther = -1;
        float maxOther = Float.NEGATIVE_INFINITY;

        for (int i = 0; i < p.length; i++) {
            if (i != correct && p[i] > maxOther) {
                maxOther = p[i];
                bestOther = i;
            }
        }

        if (1.0f - p[correct] + maxOther > 0.0f) {
            g[correct] = -1.0f;
            if (bestOther >= 0) {
                g[bestOther] = 1.0f;
            }
        }

        return g;
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) best = i;
        }
        return best;
    }

    private static void checkShapes(float[] p, float[] t) {
        if (p == null || t == null || p.length == 0
                || p.length != t.length) {
            throw new IllegalArgumentException(
                "Predictions and targets must have equal nonzero lengths");
        }
    }
}
