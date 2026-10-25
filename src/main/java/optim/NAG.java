package optim;

import java.util.Arrays;

/**
 * Nesterov Accelerated Gradient.
 */
public class NAG implements Optimizer {

    private final float learningRate;
    private final float momentum;
    private float[] velocity;

    public NAG(float learningRate) {
        this(learningRate, 0.9f);
    }

    public NAG(float learningRate, float momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    @Override
    public void update(float[] parameters, float[] gradients) {
        if (parameters == null || gradients == null ||
            parameters.length != gradients.length) {
            throw new IllegalArgumentException(
                "Parameters and gradients must have same length");
        }

        if (velocity == null || velocity.length != parameters.length) {
            velocity = new float[parameters.length];
        }

        for (int i = 0; i < parameters.length; i++) {
            velocity[i] = momentum * velocity[i] - learningRate * gradients[i];
            parameters[i] += velocity[i];
        }
    }

    public void reset() {
        if (velocity != null) {
            Arrays.fill(velocity, 0.0f);
        }
    }
}
