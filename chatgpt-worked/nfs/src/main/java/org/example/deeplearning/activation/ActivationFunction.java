package org.example.deeplearning.activation;

public interface ActivationFunction {
    double activate(double x);

    // Derivative expressed in terms of the activated output.
    double derivativeFromOutput(double output);

    static ActivationFunction sigmoid() {
        return new ActivationFunction() {
            @Override
            public double activate(double x) {
                if (x >= 0.0) {
                    return 1.0 / (1.0 + Math.exp(-x));
                }
                double e = Math.exp(x);
                return e / (1.0 + e);
            }

            @Override
            public double derivativeFromOutput(double y) {
                return y * (1.0 - y);
            }
        };
    }

    static ActivationFunction tanh() {
        return new ActivationFunction() {
            public double activate(double x) {
                return Math.tanh(x);
            }

            public double derivativeFromOutput(double y) {
                return 1.0 - y * y;
            }
        };
    }

    static ActivationFunction relu() {
        return new ActivationFunction() {
            public double activate(double x) {
                return Math.max(0.0, x);
            }

            public double derivativeFromOutput(double y) {
                return y > 0.0 ? 1.0 : 0.0;
            }
        };
    }

    static ActivationFunction leakyRelu(double alpha) {
        if (alpha < 0.0) {
            throw new IllegalArgumentException("alpha must be non-negative");
        }

        return new ActivationFunction() {
            public double activate(double x) {
                return x >= 0.0 ? x : alpha * x;
            }

            public double derivativeFromOutput(double y) {
                return y > 0.0 ? 1.0 : alpha;
            }
        };
    }

    static ActivationFunction linear() {
        return new ActivationFunction() {
            public double activate(double x) {
                return x;
            }

            public double derivativeFromOutput(double y) {
                return 1.0;
            }
        };
    }
}