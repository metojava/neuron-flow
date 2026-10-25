package initialization;

import java.util.Random;

/**
 * Strategies for choosing initial weight values. Good starting
 * values keep activations and gradients in a healthy range so
 * training starts fast and does not explode or vanish.
 */
@FunctionalInterface
public interface WeightInitializer {

    /**
     * Fills the given weight array in place.
     *
     * @param weights weights of one neuron, modified by this call
     * @param fanIn   number of inputs to the neuron
     * @param fanOut  number of neurons in this neuron's layer
     * @param rand    seeded RNG for reproducibility
     */
    void initialize(float[] weights, int fanIn, int fanOut,
                    Random rand);

    /**
     * Xavier/Glorot uniform: values in
     * [-sqrt(6/(fanIn+fanOut)), +sqrt(6/(fanIn+fanOut))].
     * Good default for sigmoid and tanh outputs.
     *
     * @return the Xavier uniform initializer
     */
    static WeightInitializer xavierUniform() {
        return (weights, fanIn, fanOut, rand) -> {
            float limit = (float) Math.sqrt(
                6.0 / (fanIn + fanOut));

            for (int i = 0; i < weights.length; i++) {
                weights[i] =
                    (rand.nextFloat() * 2.0f - 1.0f) * limit;
            }
        };
    }

    /**
     * He normal: Gaussian samples with standard deviation
     * sqrt(2/fanIn). Recommended for ReLU and its variants.
     *
     * @return the He normal initializer
     */
    static WeightInitializer heNormal() {
        return (weights, fanIn, fanOut, rand) -> {
            float stddev = (float) Math.sqrt(2.0 / fanIn);

            for (int i = 0; i < weights.length; i++) {
                weights[i] = (float) rand.nextGaussian() * stddev;
            }
        };
    }
}
