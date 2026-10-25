package optim;

/** Adam optimizer with bias-corrected first and second moments. */
public class Adam implements Optimizer {
    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private float[] firstMoment;
    private float[] secondMoment;
    private long timestep;

    public Adam(float learningRate) {
        this(learningRate, 0.9f, 0.999f, 1e-8f);
    }

    public Adam(float learningRate, float beta1,
                float beta2, float epsilon) {
        if (!Float.isFinite(learningRate) || learningRate <= 0.0f
            || !Float.isFinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f
            || !Float.isFinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f
            || !Float.isFinite(epsilon) || epsilon <= 0.0f) {
            throw new IllegalArgumentException("Invalid Adam parameters");
        }
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = 0.0f;
    }

    /**
     * Adam with weight decay (decoupled, like AdamW).
     *
     * @param learningRate step size, must be &gt; 0
     * @param beta1        first-moment decay in [0, 1)
     * @param beta2        second-moment decay in [0, 1)
     * @param epsilon      small constant, must be &gt; 0
     * @param weightDecay  decoupled weight decay coefficient, must be &gt;= 0
     * @throws IllegalArgumentException on invalid parameters
     */
    public Adam(float learningRate, float beta1, float beta2,
                float epsilon, float weightDecay) {
        if (!(learningRate > 0.0f) ||
            beta1 < 0.0f || beta1 >= 1.0f ||
            beta2 < 0.0f || beta2 >= 1.0f ||
            !(epsilon > 0.0f) || weightDecay < 0.0f) {
            throw new IllegalArgumentException("Invalid Adam parameters");
        }

        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }

    @Override
    public void update(float[] parameters, float[] gradients) {
        if (parameters == null || gradients == null
            || parameters.length != gradients.length) {
            throw new IllegalArgumentException("Parameter/gradient mismatch");
        }
        if (firstMoment == null || firstMoment.length != parameters.length) {
            firstMoment = new float[parameters.length];
            secondMoment = new float[parameters.length];
            timestep = 0;
        }

        timestep++;
        double correction1 = 1.0 - Math.pow(beta1, timestep);
        double correction2 = 1.0 - Math.pow(beta2, timestep);
        for (int i = 0; i < parameters.length; i++) {
            float gradient = gradients[i];
            firstMoment[i] = beta1 * firstMoment[i] + (1.0f - beta1) * gradient;
            secondMoment[i] = beta2 * secondMoment[i]
                + (1.0f - beta2) * gradient * gradient;
            double mean = firstMoment[i] / correction1;
            double variance = secondMoment[i] / correction2;
            parameters[i] -= learningRate * mean
                / (Math.sqrt(variance) + epsilon);
        }
    }
}