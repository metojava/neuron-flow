package optim;

/**
 * Nadam: Nesterov-accelerated Adaptive Moment Estimation.
 * Combines Adam's adaptive moments with Nesterov momentum,
 * looking ahead in the gradient direction before updating.
 * Often converges slightly faster than plain Adam.
 */
public class Nadam implements Optimizer {

    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;

    private float[] firstMoment;
    private float[] secondMoment;
    private long timestep;

    public Nadam(float learningRate) {
        this(learningRate, 0.9f, 0.999f, 1e-8f);
    }

    public Nadam(float learningRate, float beta1,
                 float beta2, float epsilon) {
        if (!(learningRate > 0.0f) ||
            beta1 < 0.0f || beta1 >= 1.0f ||
            beta2 < 0.0f || beta2 >= 1.0f ||
            !(epsilon > 0.0f)) {
            throw new IllegalArgumentException(
                "Invalid Nadam parameters");
        }
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public void update(float[] parameters, float[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException(
                "Parameter/gradient mismatch");
        }

        if (firstMoment == null ||
            firstMoment.length != parameters.length) {
            firstMoment = new float[parameters.length];
            secondMoment = new float[parameters.length];
            timestep = 0;
        }

        timestep++;
        float bc1 = 1.0f - (float) Math.pow(beta1, timestep);
        float bc2 = 1.0f - (float) Math.pow(beta2, timestep);

        for (int i = 0; i < parameters.length; i++) {
            float g = gradients[i];

            firstMoment[i] = beta1 * firstMoment[i]
                           + (1.0f - beta1) * g;

            secondMoment[i] = beta2 * secondMoment[i]
                            + (1.0f - beta2) * g * g;

            float mHat = firstMoment[i] / bc1;
            float vHat = secondMoment[i] / bc2;

            // Nesterov-style look-ahead term
            float mPrime = beta1 * mHat
                + (1.0f - beta1) * g / bc1;

            parameters[i] -= learningRate * mPrime
                / ((float) Math.sqrt(vHat) + epsilon);
        }
    }
}
