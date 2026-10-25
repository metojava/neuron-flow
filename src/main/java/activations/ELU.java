package activations;

/**
 * Exponential Linear Unit: x when x &gt;= 0, otherwise
 * alpha * (exp(x) - 1). Zero-centred like ELU, smooth, and
 * keeps a small non-zero gradient for negative inputs.
 */
public final class ELU implements ActivationFunction {

    private final float alpha;

    public ELU(float alpha) {
        if (alpha <= 0.0f) {
            throw new IllegalArgumentException(
                "alpha must be positive");
        }
        this.alpha = alpha;
    }

    public static ELU of(float alpha) {
        return new ELU(alpha);
    }

    public static ELU of() {
        return new ELU(1.0f);
    }

    @Override
    public float activate(float x) {
        return x >= 0.0f ? x : alpha * ((float) Math.exp(x) - 1.0f);
    }

    @Override
    public float derivativeFromOutput(float y) {
        return y >= 0.0f ? 1.0f : y + alpha;
    }
}
