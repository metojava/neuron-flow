package layers;

/**
 * Batch normalization layer (inference mode only). Normalizes
 * inputs using running mean and variance, then applies scale (gamma)
 * and shift (beta). Training-mode batch statistics are not implemented;
 * use pre-computed running stats for inference.
 */
public final class BatchNormLayer {

    private final int features;
    private final float[] gamma;  // scale
    private final float[] beta;   // shift
    private final float[] runningMean;
    private final float[] runningVar;
    private final float epsilon;

    /**
     * @param features    number of features
     * @param gamma       scale parameters (length features)
     * @param beta        shift parameters (length features)
     * @param runningMean running mean (length features)
     * @param runningVar  running variance (length features)
     * @param epsilon     small constant for numerical stability
     */
    public BatchNormLayer(int features, float[] gamma, float[] beta,
                          float[] runningMean, float[] runningVar,
                          float epsilon) {
        if (features <= 0) {
            throw new IllegalArgumentException("features must be positive");
        }
        this.features = features;
        this.gamma = gamma.clone();
        this.beta = beta.clone();
        this.runningMean = runningMean.clone();
        this.runningVar = runningVar.clone();
        this.epsilon = epsilon;
    }

    /** Creates a BatchNormLayer with gamma=1, beta=0, mean=0, var=1. */
    public static BatchNormLayer identity(int features) {
        float[] gamma = new float[features];
        float[] beta = new float[features];
        float[] mean = new float[features];
        float[] var = new float[features];
        for (int i = 0; i < features; i++) {
            gamma[i] = 1.0f;
            var[i] = 1.0f;
        }
        return new BatchNormLayer(features, gamma, beta, mean, var, 1e-5f);
    }

    /**
     * Forward pass (inference mode): normalizes using running stats.
     *
     * @param input input vector (length features)
     * @return normalized output
     */
    public float[] forward(float[] input) {
        float[] out = new float[features];
        for (int i = 0; i < features; i++) {
            float std = (float) Math.sqrt(runningVar[i] + epsilon);
            out[i] = gamma[i] * (input[i] - runningMean[i]) / std + beta[i];
        }
        return out;
    }
}
