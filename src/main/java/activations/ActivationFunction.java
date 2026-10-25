package activations;

/**
 * Nonlinear activation functions for one neuron.
 * Implementations are stateless and thread-safe.
 */
public interface ActivationFunction {

    /**
     * Forward activation: a = f(x).
     *
     * @param x pre-activation value
     * @return activated value
     */
    float activate(float x);

    /**
     * Derivative of f expressed in terms of the activated output
     * a = f(z), so cached neuron values can be reused.
     *
     * @param output the activated value a
     * @return f'(z)
     */
    float derivativeFromOutput(float output);

    /**
     * Logistic sigmoid, range (0, 1), numerically stable for
     * both signs of x.
     *
     * @return the sigmoid activation
     */
    static ActivationFunction sigmoid() {
        return new ActivationFunction() {
            @Override
            public float activate(float x) {
                if (x >= 0.0f) {
                    return 1.0f / (1.0f + (float) Math.exp(-x));
                }
                float e = (float) Math.exp(x);
                return e / (1.0f + e);
            }

            @Override
            public float derivativeFromOutput(float y) {
                return y * (1.0f - y);
            }
        };
    }

    /**
     * Hyperbolic tangent, range (-1, 1), zero-centred.
     *
     * @return the tanh activation
     */
    static ActivationFunction tanh() {
        return new ActivationFunction() {
            @Override
            public float activate(float x) {
                return (float) Math.tanh(x);
            }

            @Override
            public float derivativeFromOutput(float y) {
                return 1.0f - y * y;
            }
        };
    }

    /**
     * Rectified linear unit: max(0, x). Default for hidden layers.
     *
     * @return the ReLU activation
     */
    static ActivationFunction relu() {
        return new ActivationFunction() {
            @Override
            public float activate(float x) {
                return Math.max(0.0f, x);
            }

            @Override
            public float derivativeFromOutput(float y) {
                return y > 0.0f ? 1.0f : 0.0f;
            }
        };
    }

    /**
     * Leaky ReLU: x when x &gt;= 0, otherwise alpha * x.
     * Keeps a small gradient on the negative side.
     *
     * @param alpha slope for negative inputs, non-negative
     * @return the leaky ReLU activation
     * @throws IllegalArgumentException if alpha is negative
     */
    static ActivationFunction leakyRelu(float alpha) {
        if (alpha < 0.0f) {
            throw new IllegalArgumentException(
                "alpha must be non-negative");
        }

        return new ActivationFunction() {
            @Override
            public float activate(float x) {
                return x >= 0.0f ? x : alpha * x;
            }

            @Override
            public float derivativeFromOutput(float y) {
                return y > 0.0f ? 1.0f : alpha;
            }
        };
    }

    /**
     * Identity activation, used for input layers and regression
     * outputs. Derivative is always 1.
     *
     * @return the linear activation
     */
    static ActivationFunction linear() {
        return new ActivationFunction() {
            @Override
            public float activate(float x) {
                return x;
            }

            @Override
            public float derivativeFromOutput(float y) {
                return 1.0f;
            }
        };
    }

    /**
     * Derivative with respect to the pre-activation z. Defaults to
     * {@link #derivativeFromOutput}; overridden where the input is needed.
     *
     * @param z      pre-activation value
     * @param output activated value f(z)
     * @return f'(z)
     */
    default float derivative(float z, float output) {
        return derivativeFromOutput(output);
    }

    static ActivationFunction identity() {
        return linear();
    }

    static ActivationFunction elu() {
        return ELU.of();
    }

    static ActivationFunction elu(float alpha) {
        return ELU.of(alpha);
    }

    static ActivationFunction selu() {
        return SELU.of();
    }

    static ActivationFunction gelu() {
        return GELU.of();
    }

    static ActivationFunction swish() {
        return Swish.of();
    }

    static ActivationFunction silu() {
        return Swish.silu();
    }

    static ActivationFunction mish() {
        return Mish.of();
    }

    static ActivationFunction relu6() {
        return ReLU6.of();
    }

    static ActivationFunction hardswish() {
        return HardSwish.of();
    }

    static ActivationFunction prelu() {
        return PReLU.of();
    }

    static ActivationFunction prelu(float alpha) {
        return PReLU.of(alpha);
    }

    static ActivationFunction softplus() {
        return Softplus.of();
    }

    static ActivationFunction exponential() {
        return new ActivationFunction() {
            @Override
            public float activate(float x) {
                return (float) Math.exp(x);
            }

            @Override
            public float derivativeFromOutput(float y) {
                return y;
            }
        };
    }
}
