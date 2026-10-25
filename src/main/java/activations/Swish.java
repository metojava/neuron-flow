package activations;

/**
 * Swish (also called SiLU): x * sigmoid(beta * x). Smooth,
 * non-monotonic, often outperforms ReLU on deeper networks.
 * With beta = 1 this is the SiLU activation.
 */
public final class Swish implements ActivationFunction {

    private final float beta;

    public Swish(float beta) {
        if (beta <= 0.0f) {
            throw new IllegalArgumentException(
                "beta must be positive");
        }
        this.beta = beta;
    }

    public static Swish of() {
        return new Swish(1.0f);
    }

    public static Swish of(float beta) {
        return new Swish(beta);
    }

    public static Swish silu() {
        return new Swish(1.0f);
    }

    @Override
    public float activate(float x) {
        float s = sigmoid(beta * x);
        return x * s;
    }

    /**
     * Not supported: y = x*sigmoid(beta*x) does not determine the
     * derivative (some outputs come from two inputs). Use
     * {@link #derivative(float, float)} with the pre-activation.
     *
     * @throws UnsupportedOperationException always
     */
    @Override
    public float derivativeFromOutput(float y) {
        throw new UnsupportedOperationException(
            "Use derivative(z, output) for Swish");
    }

    @Override
    public float derivative(float z, float output) {
        return derivativeFromInput(z);
    }

    /**
     * Derivative expressed in terms of the pre-activation x.
     *
     * @param x pre-activation input
     * @return f'(x)
     */
    public float derivativeFromInput(float x) {
        float s = sigmoid(beta * x);
        return s + x * beta * s * (1.0f - s);
    }

    private static float sigmoid(float x) {
        if (x >= 0.0f) {
            return 1.0f / (1.0f + (float) Math.exp(-x));
        }
        float e = (float) Math.exp(x);
        return e / (1.0f + e);
    }
}
