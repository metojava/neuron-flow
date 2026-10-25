package org.example.deeplearning.loss;

public interface LossFunction {
    double loss(double[] predictions, double[] targets);

    // Gradient of loss with respect to predictions.
    double[] gradient(double[] predictions, double[] targets);

    static LossFunction meanSquaredError() {
        return new LossFunction() {
            public double loss(double[] p, double[] t) {
                checkShapes(p, t);
                double sum = 0.0;

                for (int i = 0; i < p.length; i++) {
                    double d = p[i] - t[i];
                    sum += d * d;
                }

                return sum / p.length;
            }

            public double[] gradient(double[] p, double[] t) {
                checkShapes(p, t);
                double[] g = new double[p.length];

                for (int i = 0; i < p.length; i++) {
                    g[i] = 2.0 * (p[i] - t[i]) / p.length;
                }

                return g;
            }
        };
    }

    static LossFunction binaryCrossEntropy() {
        return new LossFunction() {
            public double loss(double[] p, double[] t) {
                checkShapes(p, t);
                double sum = 0.0;
                double eps = 1e-12;

                for (int i = 0; i < p.length; i++) {
                    if (t[i] < 0.0 || t[i] > 1.0) {
                        throw new IllegalArgumentException(
                            "Binary targets must be in [0, 1]");
                    }

                    double q = Math.max(eps, Math.min(1.0 - eps, p[i]));
                    sum -= t[i] * Math.log(q)
                         + (1.0 - t[i]) * Math.log(1.0 - q);
                }

                return sum / p.length;
            }

            public double[] gradient(double[] p, double[] t) {
                checkShapes(p, t);
                double[] g = new double[p.length];
                double eps = 1e-12;

                for (int i = 0; i < p.length; i++) {
                    double q = Math.max(eps, Math.min(1.0 - eps, p[i]));
                    g[i] = (-t[i] / q + (1.0 - t[i]) / (1.0 - q))
                           / p.length;
                }

                return g;
            }
        };
    }

    private static void checkShapes(double[] p, double[] t) {
        if (p == null || t == null || p.length == 0
                || p.length != t.length) {
            throw new IllegalArgumentException(
                "Predictions and targets must have equal nonzero lengths");
        }
    }
}