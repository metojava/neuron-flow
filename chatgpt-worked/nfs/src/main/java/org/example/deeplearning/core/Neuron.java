package org.example.deeplearning.core;

import java.util.Random;

public class Neuron {
    public double[] weights;
    public double bias;
    public double value;
    public double gradient;

    public Neuron(int numInputs, Random rand) {
        this.weights = new double[numInputs];

        for (int i = 0; i < numInputs; i++) {
            this.weights[i] = rand.nextDouble() * 2.0 - 1.0;
        }

        this.bias = rand.nextDouble() * 2.0 - 1.0;
    }
}