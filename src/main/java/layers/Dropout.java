package layers;

import java.util.Random;

/**
 * Inverted dropout: during training, randomly zeroes activations
 * with probability {@code rate} and scales survivors by
 * 1/(1 - rate) so the expected value is unchanged. During
 * inference (dropout = 0 mask) it is the identity.
 * Apply between dense or conv layers as a regularizer.
 */
public class Dropout {

    private final float rate;
    private final Random random;
    private boolean training;

    /**
     * @param rate fraction of units to drop, in [0, 1)
     * @param seed RNG seed for reproducible masks
     */
    public Dropout(float rate, long seed) {
        if (rate < 0.0f || rate >= 1.0f) {
            throw new IllegalArgumentException(
                "rate must be in [0, 1)");
        }
        this.rate = rate;
        this.random = new Random(seed);
        this.training = true;
    }

    /** Enter training mode (dropout active). */
    public void setTraining(boolean training) {
        this.training = training;
    }

    /** @return true if currently in training mode */
    public boolean isTraining() {
        return training;
    }

    /**
     * Applies dropout to a vector in place during training;
     * identity during inference.
     *
     * @param activations layer outputs, modified in place
     */
    public void apply(float[] activations) {
        if (!training || rate == 0.0f) {
            return;
        }

        float scale = 1.0f / (1.0f - rate);

        for (int i = 0; i < activations.length; i++) {
            activations[i] = random.nextFloat() < rate
                ? 0.0f
                : activations[i] * scale;
        }
    }

    /**
     * Applies dropout to a batch of vectors.
     *
     * @param batch [N][features], each row modified in place
     */
    public void apply(float[][] batch) {
        for (float[] row : batch) {
            apply(row);
        }
    }
}
