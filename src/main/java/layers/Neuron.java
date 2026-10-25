package layers;

import java.util.Random;

/**
 * One artificial neuron: holds its weights (one per input),
 * its bias, the activation from the last forward pass, and the
 * error signal (delta) from the last backward pass.
 */
public class Neuron {

    /** One weight per upstream neuron (input layer: length 0). */
    public float[] weights;

    /** Additive bias applied before activation. */
    public float bias;

    /** Activation a = f(z) cached from the last forward pass. */
    public float value;

    /** Pre-activation z = bias + w.x cached from the last forward pass. */
    public float preActivation;

    /** Backpropagated error signal delta for this neuron. */
    public float gradient;

    /**
     * Creates a neuron with {@code numInputs} randomly
     * initialized weights (uniform in [-1, 1)) and a random
     * bias. The network constructor re-initializes these with
     * a scale-aware initializer.
     *
     * @param numInputs number of weights to allocate
     * @param rand      seeded RNG so runs are reproducible
     */
    public Neuron(int numInputs, Random rand) {
        this.weights = new float[numInputs];

        for (int i = 0; i < numInputs; i++) {
            this.weights[i] = rand.nextFloat() * 2.0f - 1.0f;
        }

        this.bias = rand.nextFloat() * 2.0f - 1.0f;
    }
}
