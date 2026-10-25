package activations;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks each activation's derivative(z, output) against a central
 * finite difference of activate(z). Points avoid the kinks of
 * piecewise functions (ReLU6, HardSwish, PReLU, ELU at 0).
 */
class ActivationDerivativeTest {

    private static final double H = 1e-3;

    private static List<ActivationFunction> activations() {
        return List.of(
            ActivationFunction.sigmoid(),
            ActivationFunction.tanh(),
            ActivationFunction.linear(),
            ActivationFunction.leakyRelu(0.1f),
            ActivationFunction.elu(),
            ActivationFunction.selu(),
            ActivationFunction.gelu(),
            ActivationFunction.swish(),
            ActivationFunction.silu(),
            ActivationFunction.mish(),
            ActivationFunction.softplus(),
            ActivationFunction.hardswish(),
            ActivationFunction.prelu(0.25f));
    }

    @Test
    void outputOnlyDerivativeIsRejectedForNonInvertibleActivations() {
        // The output does not determine the derivative for these,
        // so callers must use derivative(z, output).
        assertThrows(UnsupportedOperationException.class,
            () -> ActivationFunction.gelu().derivativeFromOutput(0.5f));
        assertThrows(UnsupportedOperationException.class,
            () -> ActivationFunction.swish().derivativeFromOutput(0.5f));
    }

    @Test
    void derivativeMatchesFiniteDifference() {
        float[] points = {-2.3f, -1.3f, -0.6f, 0.4f, 1.1f, 2.7f};

        for (ActivationFunction f : activations()) {
            for (float z : points) {
                float a = f.activate(z);
                double numeric =
                    (f.activate((float) (z + H))
                   - f.activate((float) (z - H))) / (2 * H);
                double analytic = f.derivative(z, a);

                assertEquals(numeric, analytic, 2e-3,
                    f.getClass().getSimpleName()
                    + " derivative wrong at z=" + z);
            }
        }
    }
}
