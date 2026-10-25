package activations;

/**
 * Hard Swish: x * ReLU6(x + 3) / 6. Used in MobileNetV3.
 */
public final class HardSwish implements ActivationFunction {

    public static HardSwish of() {
        return new HardSwish();
    }

    @Override
    public float activate(float x) {
        float relu6 = Math.min(Math.max(x + 3.0f, 0.0f), 6.0f);
        return x * relu6 / 6.0f;
    }

    @Override
    public float derivative(float z, float output) {
        return derivativeFromInput(z);
    }

    @Override
    public float derivativeFromOutput(float y) {
        // Approximate
        return 0.5f;
    }

    public float derivativeFromInput(float x) {
        if (x <= -3.0f) {
            return 0.0f;
        }
        if (x >= 3.0f) {
            return 1.0f;
        }
        return (2.0f * x + 3.0f) / 6.0f;
    }
}
