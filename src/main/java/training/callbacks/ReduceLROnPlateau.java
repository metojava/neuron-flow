package training.callbacks;

/**
 * Reduces the learning rate when a monitored metric stops improving.
 *
 * <p>Monitors validation loss by default (lower is better). When no
 * improvement is seen for {@code patience} epochs, the learning rate
 * is multiplied by {@code factor}. The new learning rate is available
 * via {@link #currentLearningRate()}.
 */
public class ReduceLROnPlateau implements TrainingCallback {

    private final float factor;
    private final int patience;
    private final boolean minimize;
    private final float minLr;
    private float currentLr;
    private double best;
    private int wait;

    /**
     * @param initialLr starting learning rate
     * @param factor    multiplier applied when plateau detected (e.g. 0.5)
     * @param patience  epochs with no improvement before reducing
     * @param minLr     lower bound for the learning rate
     * @param minimize  true if lower is better (loss), false if higher (accuracy)
     */
    public ReduceLROnPlateau(float initialLr, float factor, int patience,
                             float minLr, boolean minimize) {
        if (factor <= 0.0f || factor >= 1.0f) {
            throw new IllegalArgumentException("factor must be in (0, 1)");
        }
        if (patience < 1) {
            throw new IllegalArgumentException("patience must be >= 1");
        }
        this.currentLr = initialLr;
        this.factor = factor;
        this.patience = patience;
        this.minLr = minLr;
        this.minimize = minimize;
        this.best = minimize ? Double.MAX_VALUE : -Double.MAX_VALUE;
        this.wait = 0;
    }

    /** Monitors validation loss. */
    public ReduceLROnPlateau(float initialLr, float factor, int patience) {
        this(initialLr, factor, patience, 1e-6f, true);
    }

    @Override
    public boolean onEpochEnd(int epoch, double trainLoss,
                             double valLoss, double valAcc) {
        double monitored = minimize ? valLoss : valAcc;
        if (Double.isNaN(monitored)) {
            monitored = trainLoss;
        }

        boolean improved = minimize
            ? monitored < best - 1e-8
            : monitored > best + 1e-8;

        if (improved) {
            best = monitored;
            wait = 0;
        } else {
            wait++;
            if (wait >= patience) {
                currentLr = Math.max(minLr, currentLr * factor);
                wait = 0;
            }
        }
        return true;
    }

    /** Returns the current learning rate. */
    public float currentLearningRate() {
        return currentLr;
    }

    /** Resets the callback for a new training run. */
    public void reset() {
        best = minimize ? Double.MAX_VALUE : -Double.MAX_VALUE;
        wait = 0;
        currentLr = currentLr; // keep current
    }
}
