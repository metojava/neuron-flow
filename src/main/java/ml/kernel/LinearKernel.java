package ml.kernel;

import java.util.Objects;

/** Linear kernel: {@code K(a, b) = a . b}. Stateless and thread-safe. */
public final class LinearKernel implements Kernel {

    /** Shared immutable instance. */
    public static final LinearKernel INSTANCE = new LinearKernel();

    private LinearKernel() {
    }

    @Override
    public double evaluate(double[] a, double[] b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have equal length");
        }
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    @Override
    public String toString() {
        return "LinearKernel";
    }
}
