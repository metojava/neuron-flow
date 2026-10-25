package org.example.deeplearning.optim;

public interface Optimizer {
    void update(double[] parameters, double[] gradients);
}