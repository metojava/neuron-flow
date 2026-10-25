package optim;

/**
 * RMSProp: root mean square propagation. Keeps an exponential
 * moving average of squared gradients and divides the gradient
 * by its running RMS, which adapts the step size per parameter.
 * Defaults: rho = 0.9, epsilon = 1e-8.
 */
public class RMSProp implements Optimizer {

    private final float learningRate;
    private final float rho;
    private final float epsilon;

    private float[] cache;

    /**
     * RMSProp with standard hyper-parameters.
     *
     * @param learningRate step size, must be &gt; 0
     */
    public RMSProp(float learningRate) {
        this(learningRate, 0.9f, 1e-8f);
    }

    /**
     * RMSProp with explicit hyper-parameters.
     *
     * @param learningRate step size, must be &gt; 0
     * @param rho          decay of the running average in [0, 1)
     * @param epsilon      denominator guard, must be &gt; 0
     * @throws IllegalArgumentException on invalid parameters
     */
    public RMSProp(float learningRate, float rho,
                   float epsilon) {
        if (!(learningRate > 0.0f) ||
            rho < 0.0f || rho >= 1.0f ||
            !(epsilon > 0.0f)) {
            throw new IllegalArgumentException(
                "Invalid RMSProp parameters");
        }

        this.learningRate = learningRate;
        this.rho = rho;
        this.epsilon = epsilon;
    }

    /**
     * One RMSProp step:
     * cache = rho*cache + (1-rho)*g^2;
     * w -= lr * g / (sqrt(cache) + eps).
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

        if (cache == null || cache.length != parameters.length) {
            cache = new float[parameters.length];
        }

        for (int i = 0; i < parameters.length; i++) {
            float g = gradients[i];

            cache[i] = rho * cache[i]
                     + (1.0f - rho) * g * g;

            parameters[i] -= learningRate * g
                           / ((float) Math.sqrt(cache[i])
                              + epsilon);
        }
    }
}
