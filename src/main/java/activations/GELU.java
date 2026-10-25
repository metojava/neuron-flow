package activations;

/**
 * Gaussian Error Linear Unit: Phi(x) * x where Phi is the
 * standard normal CDF. Approximated with the tanh-based formula
 * for speed. Popular in transformer architectures.
 */
public final class GELU implements ActivationFunction {

    public static GELU of() {
        return new GELU();
    }

    @Override
    public float activate(float x) {
        return 0.5f * x * (1.0f + (float) Math.tanh(
            0.7978845608f * (x + 0.044715f * x * x * x)));
    }

    /**
     * Not supported: GELU is not invertible, so its output does not
     * determine the derivative. Use {@link #derivative(float, float)}
     * with the pre-activation.
     *
     * @throws UnsupportedOperationException always
     */
    @Override
    public float derivativeFromOutput(float y) {
        throw new UnsupportedOperationException(
            "Use derivative(z, output) for GELU");
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
        float c = 0.7978845608f;
        float x3 = x * x * x;
        float tanhArg = c * (x + 0.044715f * x3);
        float tanhVal = (float) Math.tanh(tanhArg);
        float sech2 = 1.0f - tanhVal * tanhVal;
        float dInner = c * (1.0f + 3.0f * 0.044715f * x * x);
        return 0.5f * (1.0f + tanhVal)
             + 0.5f * x * sech2 * dInner;
    }
}
