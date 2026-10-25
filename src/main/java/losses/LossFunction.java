package losses;

/**
 * Error metrics that measure how far predictions are from
 * targets, plus their gradients with respect to predictions.
 */
public interface LossFunction {

    /**
     * Scalar error for one prediction/target pair.
     *
     * @param predictions network outputs, length N
     * @param targets     desired outputs, length N
     * @return loss (smaller is better)
     */
    float loss(float[] predictions, float[] targets);

    /**
     * Gradient of the loss with respect to the predictions.
     *
     * @param predictions network outputs, length N
     * @param targets     desired outputs, length N
     * @return one partial derivative per output unit
     */
    float[] gradient(float[] predictions, float[] targets);

    /**
     * Mean squared error: mean over units of (p - t)^2.
     * Pair with a linear or sigmoid output.
     *
     * @return the MSE loss
     */
    static LossFunction meanSquaredError() {
        return new LossFunction() {
            @Override
            public float loss(float[] p, float[] t) {
                checkShapes(p, t);
                float sum = 0.0f;

                for (int i = 0; i < p.length; i++) {
                    float d = p[i] - t[i];
                    sum += d * d;
                }

                return sum / p.length;
            }

            @Override
            public float[] gradient(float[] p, float[] t) {
                checkShapes(p, t);
                float[] g = new float[p.length];

                for (int i = 0; i < p.length; i++) {
                    g[i] = 2.0f * (p[i] - t[i]) / p.length;
                }

                return g;
            }
        };
    }

    /**
     * Binary cross-entropy for targets in [0, 1].
     * Pair with a sigmoid output. Predictions are clamped away
     * from 0 and 1 so log() never returns infinity.
     *
     * @return the BCE loss
     */
    static LossFunction binaryCrossEntropy() {
        return new LossFunction() {
            @Override
            public float loss(float[] p, float[] t) {
                checkShapes(p, t);
                float sum = 0.0f;
                float eps = 1e-7f;

                for (int i = 0; i < p.length; i++) {
                    if (t[i] < 0.0f || t[i] > 1.0f) {
                        throw new IllegalArgumentException(
                            "Binary targets must be in [0, 1]");
                    }

                    float q = clamp(p[i], eps);
                    sum -= t[i] * (float) Math.log(q)
                         + (1.0f - t[i]) * (float) Math.log(1.0f - q);
                }

                return sum / p.length;
            }

            @Override
            public float[] gradient(float[] p, float[] t) {
                checkShapes(p, t);
                float[] g = new float[p.length];
                float eps = 1e-7f;

                for (int i = 0; i < p.length; i++) {
                    float q = clamp(p[i], eps);
                    g[i] = (-t[i] / q
                           + (1.0f - t[i]) / (1.0f - q))
                           / p.length;
                }

                return g;
            }
        };
    }

    /**
     * Categorical cross-entropy with a numerically stable
     * softmax built in: predictions are treated as raw logits,
     * softmaxed internally, then scored against one-hot targets.
     * The returned gradient is (softmax(logits) - target) / N,
     * which is the combined softmax + CE gradient — so pair this
     * loss with a {@code linear()} output activation (no softmax
     * neuron needed) and predict with argmax over the logits.
     *
     * @return the softmax cross-entropy loss
     */
    static LossFunction softmaxCrossEntropy() {
        return new LossFunction() {
            @Override
            public float loss(float[] p, float[] t) {
                checkShapes(p, t);

                float max = p[0];
                for (int i = 1; i < p.length; i++) {
                    if (p[i] > max) {
                        max = p[i];
                    }
                }

                float sumExp = 0.0f;
                for (int i = 0; i < p.length; i++) {
                    sumExp += (float) Math.exp(p[i] - max);
                }

                float logSumExp = max + (float) Math.log(sumExp);
                float nll = 0.0f;

                for (int i = 0; i < p.length; i++) {
                    nll += t[i] * (logSumExp - p[i]);
                }

                return nll;
            }

            @Override
            public float[] gradient(float[] p, float[] t) {
                checkShapes(p, t);
                float[] g = new float[p.length];

                float max = p[0];
                for (int i = 1; i < p.length; i++) {
                    if (p[i] > max) {
                        max = p[i];
                    }
                }

                float sumExp = 0.0f;
                for (int i = 0; i < p.length; i++) {
                    float e = (float) Math.exp(p[i] - max);
                    g[i] = e;
                    sumExp += e;
                }

                for (int i = 0; i < p.length; i++) {
                    g[i] = g[i] / sumExp - t[i];
                }

                return g;
            }
        };
    }

    static LossFunction crossEntropy() {
        return softmaxCrossEntropy();
    }

    static LossFunction categoricalCrossEntropy() {
        return softmaxCrossEntropy();
    }

    static LossFunction sparseCategoricalCrossEntropy() {
        return new LossFunction() {
            @Override
            public float loss(float[] p, float[] t) {
                checkShapes(p, t);
                // t is one-hot; same as softmax CE
                return LossFunction.softmaxCrossEntropy().loss(p, t);
            }

            @Override
            public float[] gradient(float[] p, float[] t) {
                checkShapes(p, t);
                return LossFunction.softmaxCrossEntropy().gradient(p, t);
            }
        };
    }

    static LossFunction mae() {
        return MAELoss.of();
    }

    static LossFunction l1() {
        return MAELoss.of();
    }

    private static float clamp(float value, float eps) {
        return Math.max(eps, Math.min(1.0f - eps, value));
    }

    private static void checkShapes(float[] p, float[] t) {
        if (p == null || t == null || p.length == 0
                || p.length != t.length) {
            throw new IllegalArgumentException(
                "Predictions and targets must have equal nonzero lengths");
        }
    }
}
