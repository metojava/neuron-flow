package layers;

import java.util.Random;

/**
 * A layer is a fixed-size row of neurons that all share the
 * same number of inputs. The input layer is created with zero
 * inputs per neuron: its neurons only store values.
 */
public class Layer {

    /** Neurons of this layer, in order. */
    public Neuron[] neurons;

    /**
     * Creates a fully connected layer.
     *
     * @param numNeurons         neurons in this layer
     * @param numInputsPerNeuron weights each neuron receives
     * @param rand               shared RNG for reproducibility
     */
    public Layer(int numNeurons, int numInputsPerNeuron,
                 Random rand) {
        this.neurons = new Neuron[numNeurons];

        for (int i = 0; i < numNeurons; i++) {
            this.neurons[i] =
                new Neuron(numInputsPerNeuron, rand);
        }
    }
}
