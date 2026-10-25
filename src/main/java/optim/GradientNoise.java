package optim;

import java.util.Random;

/**
 * Adds Gaussian noise to gradients during training (Neelakantan et al.).
 * Helps escape local minima and improves generalization.
 */
public final class GradientNoise {

    private GradientNoise() {
    }

    /**
     * Adds Gaussian noise to each gradient element.
     *
     * @param gradients gradient vector, modified in place
     * @param std       standard deviation of the noise
     * @param rng       random number generator
     */
    public static void addNoise(float[] gradients, float std, Random rng) {
        if (gradients == null) {
            throw new IllegalArgumentException("gradients must not be null");
        }
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] += (float) rng.nextGaussian() * std;
        }
    }

    /**
     * Adds noise with an annealing schedule: std = initialStd / (1 + decay * step).
     *
     * @param gradients  gradient vector, modified in place
     * @param initialStd initial noise standard deviation
     * @param decay       decay rate
     * @param step       current training step
     * @param rng        random number generator
     */
    public static void addNoiseAnnealed(float[] gradients, float initialStd,
                                        float decay, int step, Random rng) {
        float std = initialStd / (1.0f + decay * step);
        addNoise(gradients, std, rng);
    }
}
