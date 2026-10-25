package layers;

import java.util.Random;

/**
 * Dropout layer: randomly zeroes a fraction of inputs during training
 * and scales the rest by 1/(1-rate) so the expected sum is unchanged.
 * During inference (eval mode) the input passes through unchanged.
 */
public final class DropoutLayer {

    private final float rate;
    private final Random rng;
    private boolean training = true;
    private boolean[] mask;

    /**
     * @param rate dropout probability in [0, 1)
     * @param seed RNG seed for reproducibility
     * @throws IllegalArgumentException on invalid rate
     */
    public DropoutLayer(float rate, long seed) {
        if (rate < 0.0f || rate >= 1.0f) {
            throw new IllegalArgumentException("rate must be in [0, 1)");
        }
        this.rate = rate;
        this.rng = new Random(seed);
    }

    /** Sets training mode (dropout active) or eval mode (passthrough). */
    public void setTraining(boolean training) {
        this.training = training;
    }

    /** Returns true if dropout is active. */
    public boolean isTraining() {
        return training;
    }

    /**
     * Forward pass. During training, zeroes elements with probability
     * {@code rate} and scales the rest. During eval, returns a copy.
     *
     * @param input input vector
     * @return output vector (new array)
     */
    public float[] forward(float[] input) {
        float[] out = new float[input.length];
        if (!training || rate == 0.0f) {
            System.arraycopy(input, 0, out, 0, input.length);
            mask = null;
            return out;
        }
        mask = new boolean[input.length];
        float scale = 1.0f / (1.0f - rate);
        for (int i = 0; i < input.length; i++) {
            if (rng.nextFloat() >= rate) {
                mask[i] = true;
                out[i] = input[i] * scale;
            } else {
                mask[i] = false;
                out[i] = 0.0f;
            }
        }
        return out;
    }

    /**
     * Backward pass: passes gradient through where the mask was true,
     * zeroes it where the mask was false.
     *
     * @param grad gradient w.r.t. output
     * @return gradient w.r.t. input
     */
    public float[] backward(float[] grad) {
        float[] out = new float[grad.length];
        if (mask == null || !training) {
            System.arraycopy(grad, 0, out, 0, grad.length);
            return out;
        }
        float scale = 1.0f / (1.0f - rate);
        for (int i = 0; i < grad.length; i++) {
            out[i] = mask[i] ? grad[i] * scale : 0.0f;
        }
        return out;
    }
}
