package org.example.deeplearning.optim;

/** RMSProp with an exponential moving average of squared gradients. */
public final class RMSProp implements Optimizer {
    private final double learningRate, decay, epsilon;
    private double[] meanSquare;
    public RMSProp(double learningRate) { this(learningRate, 0.9, 1e-8); }
    public RMSProp(double learningRate, double decay, double epsilon) {
        if (!Double.isFinite(learningRate) || learningRate <= 0 || !Double.isFinite(decay)
                || decay < 0 || decay >= 1 || !Double.isFinite(epsilon) || epsilon <= 0)
            throw new IllegalArgumentException("Invalid RMSProp parameters");
        this.learningRate = learningRate; this.decay = decay; this.epsilon = epsilon;
    }
    @Override public void update(double[] parameters, double[] gradients) {
        if (parameters == null || gradients == null || parameters.length != gradients.length)
            throw new IllegalArgumentException("Parameter/gradient mismatch");
        if (meanSquare == null || meanSquare.length != parameters.length)
            meanSquare = new double[parameters.length];
        for (int i = 0; i < parameters.length; i++) {
            meanSquare[i] = decay * meanSquare[i] + (1 - decay) * gradients[i] * gradients[i];
            parameters[i] -= learningRate * gradients[i] / (Math.sqrt(meanSquare[i]) + epsilon);
        }
    }
}
