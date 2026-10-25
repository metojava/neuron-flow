package activations;

/**
 * Mish: x * tanh(softplus(x)). Self-regularizing, smooth activation.
 */
public final class Mish implements ActivationFunction {

    public static Mish of() {
        return new Mish();
    }

    @Override
    public float activate(float x) {
        float sp = softplus(x);
        return x * (float) Math.tanh(sp);
    }

    @Override
    public float derivative(float z, float output) {
        return derivativeFromInput(z);
    }

    @Override
    public float derivativeFromOutput(float y) {
        // Fallback approximation
        return 0.5f + y * 0.1f;
    }

    public float derivativeFromInput(float x) {
        float sp = softplus(x);
        float tanhSp = (float) Math.tanh(sp);
        float omega = 4.0f * (x + 1.0f) * (float) Math.exp(2.0f * x + sp);
        float delta = (float) Math.exp(3.0f * x + 2.0f * sp)
            + 4.0f * (float) Math.exp(2.0f * x + sp)
            + (4.0f + 6.0f * x) * (float) Math.exp(x + sp)
            + 4.0f * (x + 1.0f);
        // Simpler stable form
        float sigX = sigmoid(x);
        float omega2 = 2.0f * (float) Math.exp(3.0f * x)
            + 4.0f * (float) Math.exp(2.0f * x)
            + (6.0f * x + 4.0f) * (float) Math.exp(x)
            + 4.0f * (x + 1.0f);
        float delta2 = (float) Math.exp(3.0f * x)
            + 4.0f * (float) Math.exp(2.0f * x)
            + (6.0f * x + 4.0f) * (float) Math.exp(x)
            + 4.0f;
        // Use standard formula
        float f = activate(x);
        return (float) Math.tanh(sp)
            + x * sigmoid(x) * (1.0f - tanhSp * tanhSp);
    }

    private static float softplus(float x) {
        if (x > 20.0f) {
            return x;
        }
        if (x < -20.0f) {
            return (float) Math.exp(x);
        }
        return (float) Math.log1p((float) Math.exp(x));
    }

    private static float sigmoid(float x) {
        if (x >= 0.0f) {
            return 1.0f / (1.0f + (float) Math.exp(-x));
        }
        float e = (float) Math.exp(x);
        return e / (1.0f + e);
    }
}
