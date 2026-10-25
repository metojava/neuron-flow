package layers;

/**
 * Batch normalization: normalizes each feature across the batch
 * to zero mean and unit variance, then applies a fixed affine
 * transform (scale gamma, shift beta). Keeps separate running
 * statistics for each feature for inference.
 * Works on a full mini-batch at once.
 */
public class BatchNormalization {

    private final float gamma;
    private final float beta;
    private final float epsilon;
    private final float momentum;

    /** Running estimates for inference. */
    private float[] runningMean;
    private float[] runningVar;
    private boolean initialized;

    /** Per-feature scale and shift; created from the scalar gamma and beta on first use. */
    private float[] gammaVec;
    private float[] betaVec;

    public BatchNormalization(float gamma, float beta,
                              float epsilon, float momentum) {
        if (!Float.isFinite(gamma) || !Float.isFinite(beta)
            || !Float.isFinite(epsilon) || epsilon <= 0.0f
            || !Float.isFinite(momentum) || momentum < 0.0f
            || momentum >= 1.0f) {
            throw new IllegalArgumentException("Invalid batch normalization parameters");
        }
        this.gamma = gamma;
        this.beta = beta;
        this.epsilon = epsilon;
        this.momentum = momentum;
        this.initialized = false;
    }

    public static BatchNormalization of() {
        return new BatchNormalization(1.0f, 0.0f,
            1e-5f, 0.1f);
    }

    /**
     * Normalizes a batch in place using batch statistics, and
     * updates the running estimates for later inference use.
     *
     * @param batch [N][features], each row modified in place
     */
    public void normalize(float[][] batch) {
        if (batch == null || batch.length == 0) {
            throw new IllegalArgumentException(
                "Batch must be nonempty");
        }

        int n = batch.length;
        if (batch[0] == null || batch[0].length == 0) {
            throw new IllegalArgumentException("Batch rows must be nonempty");
        }
        int features = batch[0].length;
        for (float[] row : batch) {
            if (row == null || row.length != features) {
                throw new IllegalArgumentException("Batch rows must have equal dimensions");
            }
            for (float value : row) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("Batch values must be finite");
                }
            }
        }
        if (runningMean == null) {
            runningMean = new float[features];
            runningVar = new float[features];
        } else if (runningMean.length != features) {
            throw new IllegalArgumentException("Feature count changed");
        }
        ensureAffine(features);

        for (int f = 0; f < features; f++) {
            float mean = 0.0f;
            for (int i = 0; i < n; i++) {
                mean += batch[i][f];
            }
            mean /= n;

            float variance = 0.0f;
            for (int i = 0; i < n; i++) {
                float d = batch[i][f] - mean;
                variance += d * d;
            }
            variance /= n;

            float invStd =
                1.0f / (float) Math.sqrt(variance + epsilon);

            for (int i = 0; i < n; i++) {
                batch[i][f] = gammaVec[f]
                    * (batch[i][f] - mean) * invStd + betaVec[f];
            }

            if (!initialized) {
                runningMean[f] = mean;
                runningVar[f] = variance;
            } else {
                runningMean[f] = momentum * runningMean[f]
                            + (1.0f - momentum) * mean;
                runningVar[f]  = momentum * runningVar[f]
                            + (1.0f - momentum) * variance;
            }
        }
        initialized = true;
    }

    /**
     * Normalizes a single vector using the stored running
     * statistics (for inference after training).
     *
     * @param x one example, modified in place
     */
    public void normalizeInference(float[] x) {
        if (!initialized || x == null || x.length != runningMean.length) {
            throw new IllegalArgumentException("Fit on a batch with matching features first");
        }
        for (int f = 0; f < x.length; f++) {
            if (!Float.isFinite(x[f])) {
                throw new IllegalArgumentException("Input values must be finite");
            }
            float invStd = 1.0f / (float) Math.sqrt(
                runningVar[f] + epsilon);
            x[f] = gammaVec[f] * (x[f] - runningMean[f]) * invStd + betaVec[f];
        }
    }

    /**
     * Loads per-feature scale, shift and running statistics, for example from a
     * PyTorch BatchNorm1d checkpoint. The layer is then ready for inference, and
     * further training batches update the loaded statistics with momentum.
     * The epsilon given at construction is kept.
     *
     * @param gammaValues  scale per feature
     * @param betaValues   shift per feature
     * @param meanValues   running mean per feature
     * @param varValues    running variance per feature
     * @throws IllegalArgumentException on empty or mismatched arrays
     */
    public void loadParameters(float[] gammaValues, float[] betaValues,
                               float[] meanValues, float[] varValues) {
        int f = gammaValues.length;
        if (f == 0 || betaValues.length != f || meanValues.length != f || varValues.length != f) {
            throw new IllegalArgumentException("All parameter arrays must have the same nonzero length");
        }
        gammaVec = gammaValues.clone();
        betaVec = betaValues.clone();
        runningMean = meanValues.clone();
        runningVar = varValues.clone();
        initialized = true;
    }

    /** @return number of features, or 0 if not yet known */
    public int features() {
        return runningMean == null ? 0 : runningMean.length;
    }

    private void ensureAffine(int features) {
        if (gammaVec == null) {
            gammaVec = new float[features];
            betaVec = new float[features];
            java.util.Arrays.fill(gammaVec, gamma);
            java.util.Arrays.fill(betaVec, beta);
        }
        if (gammaVec.length != features) {
            throw new IllegalArgumentException("Feature count changed");
        }
    }
}
