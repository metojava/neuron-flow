package org.example.deeplearning.optim;

import java.util.Arrays;

public class SGD implements Optimizer {
    private final double learningRate;
    private final double momentum;
    private double[] velocity;

    public SGD(double learningRate) {
        this(learningRate, 0.0);
    }

    public SGD(double learningRate, double momentum) {
        if (!(learningRate > 0.0) ||
            momentum < 0.0 || momentum >= 1.0) {
            throw new IllegalArgumentException("Invalid SGD parameters");
        }

        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    @Override
    public void update(double[] parameters, double[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException("Parameter/gradient mismatch");
        }

        if (velocity == null || velocity.length != parameters.length) {
            velocity = new double[parameters.length];
        }

        for (int i = 0; i < parameters.length; i++) {
            velocity[i] = momentum * velocity[i] + gradients[i];
            parameters[i] -= learningRate * velocity[i];
        }
    }
}