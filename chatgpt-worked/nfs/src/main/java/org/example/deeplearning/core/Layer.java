package org.example.deeplearning.core;

import java.util.Random;

public class Layer {
    public Neuron[] neurons;

    public Layer(int numNeurons, int numInputsPerNeuron, Random rand) {
        this.neurons = new Neuron[numNeurons];

        for (int i = 0; i < numNeurons; i++) {
            this.neurons[i] = new Neuron(numInputsPerNeuron, rand);
        }
    }
}