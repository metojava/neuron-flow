package activations;

/**
 * ReLU6: min(max(0, x), 6). Common in mobile networks (MobileNet).
 */
public final class ReLU6 implements ActivationFunction {

    private static final float CAP = 6.0f;

    public static ReLU6 of() {
        return new ReLU6();
    }

    @Override
    public float activate(float x) {
        if (x <= 0.0f) {
            return 0.0f;
        }
        if (x >= CAP) {
            return CAP;
        }
        return x;
    }

    @Override
    public float derivative(float z, float output) {
        if (z > 0.0f && z < CAP) {
            return 1.0f;
        }
        return 0.0f;
    }

    @Override
    public float derivativeFromOutput(float y) {
        if (y > 0.0f && y < CAP) {
            return 1.0f;
        }
        return 0.0f;
    }
}
