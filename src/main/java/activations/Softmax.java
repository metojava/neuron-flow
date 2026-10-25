package activations;

/**
 * Utilities for the softmax function, which maps a vector of
 * logits to a probability distribution. Softmax is a vector
 * operation (unlike the per-neuron activations in
 * {@link ActivationFunction}), so it lives here as a helper.
 */
public final class Softmax {

    private Softmax() {
    }

    /**
     * Numerically stable softmax over a vector of logits.
     * The maximum logit is subtracted before exponentiating to
     * avoid overflow for large inputs.
     *
     * @param logits raw scores, length N
     * @return probabilities of length N, each in (0, 1),
     *         summing to 1
     * @throws IllegalArgumentException if logits is null or empty
     */
    public static float[] apply(float[] logits) {
        if (logits == null || logits.length == 0) {
            throw new IllegalArgumentException(
                "Logits must not be empty");
        }

        float max = logits[0];

        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > max) {
                max = logits[i];
            }
        }

        float sum = 0.0f;
        float[] result = new float[logits.length];

        for (int i = 0; i < logits.length; i++) {
            result[i] = (float) Math.exp(logits[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < logits.length; i++) {
            result[i] /= sum;
        }

        return result;
    }
}
