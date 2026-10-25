package activations;

/**
 * Softplus: log(1 + exp(x)). Smooth approximation of ReLU.
 * Always positive, differentiable everywhere.
 */
public final class Softplus implements ActivationFunction {

    public static Softplus of() {
        return new Softplus();
    }

    @Override
    public float activate(float x) {
        // Numerically stable: max(0,x) + log1p(exp(-|x|))
        return Math.max(0.0f, x)
             + (float) Math.log1p((float) Math.exp(-Math.abs(x)));
    }

    @Override
    public float derivativeFromOutput(float y) {
        // f'(x) = sigmoid(x); y = softplus(x)
        // sigmoid(x) = 1 - exp(-softplus(x)) = 1 - exp(-y)
        return 1.0f - (float) Math.exp(-y);
    }
}
