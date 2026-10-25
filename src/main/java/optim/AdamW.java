package optim;

import java.util.Arrays;

/**
 * AdamW optimizer: Adam with decoupled weight decay.
 */
public class AdamW implements Optimizer {

    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private float[] m;
    private float[] v;
    private int timestep;

    public AdamW(float learningRate) {
        this(learningRate, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    public AdamW(float learningRate, float beta1, float beta2,
                float epsilon, float weightDecay) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.timestep = 0;
    }

    @Override
    public void update(float[] parameters, float[] gradients) {
        if (parameters == null || gradients == null ||
            parameters.length != gradients.length) {
            throw new IllegalArgumentException(
                "Parameters and gradients must have same length");
        }

        if (m == null || m.length != parameters.length) {
            m = new float[parameters.length];
            v = new float[parameters.length];
        }

        timestep++;

        float mHatCorrection = 1.0f - (float) Math.pow(beta1, timestep);
        float vHatCorrection = 1.0f - (float) Math.pow(beta2, timestep);

        for (int i = 0; i < parameters.length; i++) {
            // Decoupled weight decay
            if (weightDecay != 0.0f) {
                parameters[i] -= learningRate * weightDecay * parameters[i];
            }

            m[i] = beta1 * m[i] + (1.0f - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * gradients[i] * gradients[i];

            float mHat = m[i] / mHatCorrection;
            float vHat = v[i] / vHatCorrection;

            parameters[i] -= learningRate * mHat
                / ((float) Math.sqrt(vHat) + epsilon);
        }
    }

    public void reset() {
        Arrays.fill(m, 0.0f);
        Arrays.fill(v, 0.0f);
        timestep = 0;
    }
}
