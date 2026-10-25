package transformers;

import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * AdamW optimizer with weight decay decoupled from gradient updates.
 * Matches Hugging Face / PyTorch AdamW behavior.
 */
public final class AdamW {

    private final List<Tensor> params;
    private float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final List<float[]> m;
    private final List<float[]> v;
    private int step;

    public AdamW(List<Tensor> params, float lr, float beta1, float beta2,
                  float eps, float weightDecay) {
        this.params = new ArrayList<>(params);
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.m = new ArrayList<>();
        this.v = new ArrayList<>();
        this.step = 0;

        // Initialize moment buffers
        for (Tensor p : params) {
            int size = p.numel();
            m.add(new float[size]);
            v.add(new float[size]);
        }
    }

    /**
     * Performs a single optimization step.
     *
     * @param grads gradients for each parameter (same order as params)
     */
    public void step(List<float[]> grads) {
        if (grads.size() != params.size()) {
            throw new IllegalArgumentException("Gradient count mismatch");
        }
        step++;
        float biasCorrection1 = 1.0f - (float) Math.pow(beta1, step);
        float biasCorrection2 = 1.0f - (float) Math.pow(beta2, step);

        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            float[] g = grads.get(i);
            float[] m_i = m.get(i);
            float[] v_i = v.get(i);
            float[] pData = p.toFlat();

            for (int j = 0; j < pData.length; j++) {
                // Update biased first moment estimate
                m_i[j] = beta1 * m_i[j] + (1 - beta1) * g[j];
                // Update biased second raw moment estimate
                v_i[j] = beta2 * v_i[j] + (1 - beta2) * g[j] * g[j];

                // Bias correction
                float mHat = m_i[j] / biasCorrection1;
                float vHat = v_i[j] / biasCorrection2;

                // AdamW update: weight decay applied directly to parameters
                float update = mHat / ((float) Math.sqrt(vHat) + eps);
                pData[j] -= lr * (update + weightDecay * pData[j]);
            }
        }
    }

    public void zeroGrad() {
        for (Tensor p : params) {
            p.zeroGrad();
        }
    }

    public int step() { return step; }
    public float lr() { return lr; }
    public void setLr(float lr) { this.lr = lr; }
}