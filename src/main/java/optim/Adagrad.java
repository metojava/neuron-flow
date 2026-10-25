package optim;

/**
 * Adagrad: adapts the learning rate per parameter by dividing by
 * the square root of the sum of all past squared gradients.
 * Good for sparse features; the learning rate shrinks over time.
 */
public class Adagrad implements Optimizer {

    private final float learningRate;
    private final float epsilon;

    private float[] accumulator;

    public Adagrad(float learningRate) {
        this(learningRate, 1e-8f);
    }

    public Adagrad(float learningRate, float epsilon) {
        if (!(learningRate > 0.0f) || !(epsilon > 0.0f)) {
            throw new IllegalArgumentException(
                "learningRate and epsilon must be positive");
        }
        this.learningRate = learningRate;
        this.epsilon = epsilon;
    }

    @Override
    public void update(float[] parameters, float[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException(
                "Parameter/gradient mismatch");
        }

        if (accumulator == null ||
            accumulator.length != parameters.length) {
            accumulator = new float[parameters.length];
        }

        for (int i = 0; i < parameters.length; i++) {
            float g = gradients[i];
            accumulator[i] += g * g;
            parameters[i] -= learningRate * g
                / ((float) Math.sqrt(accumulator[i]) + epsilon);
        }
    }
}
