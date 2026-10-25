package ml.kernel;

/**
 * Mercer kernel used by kernelised learners. Implementations must be
 * symmetric and positive semi-definite; both kernels shipped with the library
 * satisfy these requirements.
 */
@FunctionalInterface
public interface Kernel {

    /** Evaluates {@code K(a, b)}. */
    double evaluate(double[] a, double[] b);

    /** Linear kernel {@code a . b}. */
    static Kernel linear() {
        return LinearKernel.INSTANCE;
    }

    /** Radial-basis-function kernel with the supplied bandwidth parameter. */
    static Kernel rbf(double gamma) {
        return new RbfKernel(gamma);
    }
}
