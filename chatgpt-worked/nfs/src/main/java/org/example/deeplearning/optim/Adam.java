package org.example.deeplearning.optim;

public class Adam implements Optimizer {
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    private double[] firstMoment;
    private double[] secondMoment;
    private long timestep;

    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public Adam(double learningRate, double beta1,
                double beta2, double epsilon) {
        if (!(learningRate > 0.0) ||
            beta1 < 0.0 || beta1 >= 1.0 ||
            beta2 < 0.0 || beta2 >= 1.0 ||
            !(epsilon > 0.0)) {
            throw new IllegalArgumentException("Invalid Adam parameters");
        }

        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public void update(double[] parameters, double[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException("Parameter/gradient mismatch");
        }

        if (firstMoment == null ||
            firstMoment.length != parameters.length) {
            firstMoment = new double[parameters.length];
            secondMoment = new double[parameters.length];
            timestep = 0;
        }

        timestep++;
        double correction1 = 1.0 - Math.pow(beta1, timestep);
        double correction2 = 1.0 - Math.pow(beta2, timestep);

        for (int i = 0; i < parameters.length; i++) {
            double g = gradients[i];

            firstMoment[i] = beta1 * firstMoment[i]
                           + (1.0 - beta1) * g;

            secondMoment[i] = beta2 * secondMoment[i]
                            + (1.0 - beta2) * g * g;

            double mHat = firstMoment[i] / correction1;
            double vHat = secondMoment[i] / correction2;

            parameters[i] -= learningRate * mHat
                           / (Math.sqrt(vHat) + epsilon);
        }
    }
}