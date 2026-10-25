package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.Random;

/**
 * Dropout layer for training-time regularization.
 */
public final class Dropout {

    private final float rate;
    private final Random random;

    /**
     * @param rate probability of zeroing an element (0 = no dropout)
     * @param seed RNG seed
     */
    public Dropout(float rate, long seed) {
        if (rate < 0 || rate >= 1) {
            throw new IllegalArgumentException("rate must be in [0, 1)");
        }
        this.rate = rate;
        this.random = new Random(seed);
    }

    /**
     * Applies dropout during training; returns input unchanged during eval.
     *
     * @param x     input tensor
     * @param train true during training, false for inference
     * @return output with same shape as input
     */
    public Tensor forward(Tensor x, boolean train) {
        if (!train || rate == 0.0f) return x;

        float[] data = x.data();
        float[] out = new float[data.length];
        float scale = 1.0f / (1.0f - rate);
        for (int i = 0; i < data.length; i++) {
            if (random.nextFloat() < rate) {
                out[i] = 0.0f;
            } else {
                out[i] = data[i] * scale;
            }
        }
        return Tensor.of(out, x.shape());
    }
}