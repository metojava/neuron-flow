package activations;

/**
 * Parametric ReLU: f(x) = x if x > 0 else alpha * x.
 * Alpha is learnable per channel in full implementations; here a
 * fixed alpha version is provided.
 */
public final class PReLU implements ActivationFunction {

    private final float alpha;

    public PReLU(float alpha) {
        if (alpha < 0.0f) {
            throw new IllegalArgumentException("alpha must be non-negative");
        }
        this.alpha = alpha;
    }

    public static PReLU of(float alpha) {
        return new PReLU(alpha);
    }

    public static PReLU of() {
        return new PReLU(0.25f);
    }

    @Override
    public float activate(float x) {
        return x >= 0.0f ? x : alpha * x;
    }

    @Override
    public float derivative(float z, float output) {
        return z > 0.0f ? 1.0f : alpha;
    }

    @Override
    public float derivativeFromOutput(float y) {
        return y > 0.0f ? 1.0f : alpha;
    }
}
