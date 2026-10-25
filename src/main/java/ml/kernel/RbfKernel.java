package ml.kernel;

import java.util.Objects;

/**
 * Radial-basis-function (Gaussian) kernel:
 * {@code K(a, b) = exp(-gamma * ||a - b||^2)}.
 *
 * <p>Larger {@code gamma} values produce narrower, more local decision
 * boundaries. A common heuristic for standardised features is
 * {@code gamma = 1 / numberOfFeatures}.
 */
public final class RbfKernel implements Kernel {

    private final double gamma;

    public RbfKernel(double gamma) {
        if (!Double.isFinite(gamma) || gamma <= 0.0) {
            throw new IllegalArgumentException("gamma must be positive and finite");
        }
        this.gamma = gamma;
    }

    /** Gamma heuristic for {@code features} standardised features. */
    public static RbfKernel withDefaultGamma(int features) {
        if (features <= 0) {
            throw new IllegalArgumentException("features must be positive");
        }
        return new RbfKernel(1.0 / features);
    }

    public double gamma() {
        return gamma;
    }

    @Override
    public double evaluate(double[] a, double[] b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have equal length");
        }
        double squared = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            squared += diff * diff;
        }
        return Math.exp(-gamma * squared);
    }

    @Override
    public String toString() {
        return "RbfKernel{gamma=" + gamma + '}';
    }
}
