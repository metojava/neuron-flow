package optim;

/**
 * Stochastic gradient descent with optional momentum:
 * v = momentum * v + g;  w = w - learningRate * v.
 * With momentum 0 this is plain SGD.
 */
public class SGD implements Optimizer {

    private final float learningRate;
    private final float momentum;
    private final float weightDecay;
    private float[] velocity;

    /**
     * Plain SGD without momentum.
     *
     * @param learningRate step size, must be &gt; 0
     */
    public SGD(float learningRate) {
        this(learningRate, 0.0f);
    }

    /**
     * SGD with momentum.
     *
     * @param learningRate step size, must be &gt; 0
     * @param momentum     velocity retention in [0, 1)
     * @throws IllegalArgumentException on invalid parameters
     */
    public SGD(float learningRate, float momentum) {
        this(learningRate, momentum, 0.0f);
    }

    /**
     * SGD with momentum and weight decay.
     *
     * @param learningRate step size, must be &gt; 0
     * @param momentum     velocity retention in [0, 1)
     * @param weightDecay  L2 penalty coefficient, must be &gt;= 0
     * @throws IllegalArgumentException on invalid parameters
     */
    public SGD(float learningRate, float momentum, float weightDecay) {
        if (!(learningRate > 0.0f) ||
            momentum < 0.0f || momentum >= 1.0f ||
            weightDecay < 0.0f) {
            throw new IllegalArgumentException(
                "Invalid SGD parameters");
        }

        this.learningRate = learningRate;
        this.momentum = momentum;
        this.weightDecay = weightDecay;
    }

    /**
     * Updates velocity and parameters element-wise.
     *
     * @param parameters weights, updated in place
     * @param gradients  their gradients; same length required
     * @throws IllegalArgumentException on length mismatch
     */
    @Override
    public void update(float[] parameters, float[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException(
                "Parameter/gradient mismatch");
        }

        if (velocity == null ||
            velocity.length != parameters.length) {
            velocity = new float[parameters.length];
        }

        for (int i = 0; i < parameters.length; i++) {
            float g = gradients[i];
            if (weightDecay != 0.0f) {
                g += weightDecay * parameters[i];
            }
            velocity[i] = momentum * velocity[i] + g;
            parameters[i] -= learningRate * velocity[i];
        }
    }
}
