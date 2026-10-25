package losses;

/**
 * Categorical cross-entropy without built-in softmax: expects
 * the network output to already be a probability distribution
 * (e.g. a softmax activation on the output layer). Use this when
 * you want an explicit softmax neuron; otherwise prefer
 * {@code LossFunction.softmaxCrossEntropy()} which fuses softmax
 * and CE for numerical stability.
 */
public final class CategoricalCrossEntropyLoss
        implements LossFunction {

    public static CategoricalCrossEntropyLoss of() {
        return new CategoricalCrossEntropyLoss();
    }

    @Override
    public float loss(float[] p, float[] t) {
        checkShapes(p, t);
        float sum = 0.0f;
        float eps = 1e-7f;

        for (int i = 0; i < p.length; i++) {
            float pi = Math.max(p[i], eps);
            sum -= t[i] * (float) Math.log(pi);
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
