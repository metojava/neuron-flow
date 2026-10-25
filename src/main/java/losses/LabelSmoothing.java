package losses;

/**
 * Label smoothing for cross-entropy loss. Converts hard one-hot targets
 * into soft targets: (1 - epsilon) * onehot + epsilon / numClasses.
 *
 * <p>Usage: smooth the targets before computing the loss:
 * <pre>
 *   float[] soft = LabelSmoothing.smooth(oneHot, 0.1f, 10);
 *   float loss = LossFunction.softmaxCrossEntropy().loss(logits, soft);
 * </pre>
 */
public final class LabelSmoothing {

    private LabelSmoothing() {
    }

    /**
     * Smooths a one-hot target vector.
     *
     * @param oneHot     one-hot target (sums to 1)
     * @param epsilon    smoothing factor in [0, 1)
     * @param numClasses number of classes
     * @return smoothed target vector
     * @throws IllegalArgumentException on invalid inputs
     */
    public static float[] smooth(float[] oneHot, float epsilon,
                                 int numClasses) {
        if (oneHot == null || oneHot.length != numClasses) {
            throw new IllegalArgumentException(
                "oneHot must have length numClasses");
        }
        if (epsilon < 0.0f || epsilon >= 1.0f) {
            throw new IllegalArgumentException(
                "epsilon must be in [0, 1)");
        }
        if (numClasses < 2) {
            throw new IllegalArgumentException(
                "numClasses must be >= 2");
        }

        float[] result = new float[numClasses];
        float uniform = epsilon / numClasses;
        for (int i = 0; i < numClasses; i++) {
            result[i] = uniform + (1.0f - epsilon) * oneHot[i];
        }
        return result;
    }

    /**
     * Smooths a batch of one-hot targets.
     *
     * @param targets    [batch][classes] one-hot targets
     * @param epsilon    smoothing factor in [0, 1)
     * @return smoothed targets, same shape
     */
    public static float[][] smoothBatch(float[][] targets, float epsilon) {
        if (targets == null || targets.length == 0) {
            throw new IllegalArgumentException("targets must not be empty");
        }
        int numClasses = targets[0].length;
        float[][] result = new float[targets.length][];
        for (int i = 0; i < targets.length; i++) {
            result[i] = smooth(targets[i], epsilon, numClasses);
        }
        return result;
    }
}
