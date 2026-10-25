package activations;

/**
 * Scaled Exponential Linear Unit (SELU): self-normalizing
 * activation. With the standard alpha=1.67326324 and
 * scale=1.050701, networks of SELU layers approximately preserve
 * zero mean and unit variance across layers, so batch norm is
 * often unnecessary.
 */
public final class SELU implements ActivationFunction {

    private static final float ALPHA = 1.67326324f;
    private static final float SCALE = 1.0507010f;

    public static SELU of() {
        return new SELU();
    }

    @Override
    public float activate(float x) {
        return SCALE * (x >= 0.0f ? x
            : ALPHA * ((float) Math.exp(x) - 1.0f));
    }

    @Override
    public float derivativeFromOutput(float y) {
        return y >= 0.0f ? SCALE
            : y + SCALE * ALPHA;
    }
}
