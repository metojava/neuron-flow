package transformers;

import autograd.Tensor;

import java.util.Random;

/**
 * Weight initialization utilities matching common transformer schemes.
 */
public final class Init {

    private Init() {}

    /**
     * Normal initialization with std = 0.02 (BERT / GPT-2 style).
     */
    public static void normal(Tensor t, float std, long seed) {
        float[] data = t.data();
        Random r = new Random(seed);
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (r.nextGaussian() * std);
        }
    }

    /**
     * Xavier / Glorot uniform initialization.
     */
    public static void xavierUniform(Tensor t, long seed) {
        int fanIn = t.size(0);
        int fanOut = t.size(t.ndim() - 1);
        float limit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        float[] data = t.data();
        Random r = new Random(seed);
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (r.nextDouble() * 2 * limit - limit);
        }
    }

    /**
     * Xavier / Glorot normal initialization.
     */
    public static void xavierNormal(Tensor t, long seed) {
        int fanIn = t.size(0);
        int fanOut = t.size(t.ndim() - 1);
        float std = (float) Math.sqrt(2.0 / (fanIn + fanOut));
        normal(t, std, seed);
    }

    /**
     * Kaiming / He normal initialization (for ReLU/GELU).
     */
    public static void kaimingNormal(Tensor t, long seed) {
        int fanIn = t.size(0);
        float std = (float) Math.sqrt(2.0 / fanIn);
        normal(t, std, seed);
    }

    /**
     * Truncated normal (approximate): reject samples > 2*std.
     * Matches TensorFlow/PyTorch truncated_normal_.
     */
    public static void truncatedNormal(Tensor t, float std, long seed) {
        float[] data = t.data();
        Random r = new Random(seed);
        for (int i = 0; i < data.length; i++) {
            float val;
            do {
                val = (float) (r.nextGaussian() * std);
            } while (Math.abs(val) > 2 * std);
            data[i] = val;
        }
    }

    /**
     * Zero initialization.
     */
    public static void zeros(Tensor t) {
        float[] data = t.data();
        for (int i = 0; i < data.length; i++) data[i] = 0.0f;
    }

    /**
     * Ones initialization.
     */
    public static void ones(Tensor t) {
        float[] data = t.data();
        for (int i = 0; i < data.length; i++) data[i] = 1.0f;
    }
}