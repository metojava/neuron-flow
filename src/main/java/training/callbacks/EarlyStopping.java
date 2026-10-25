package training.callbacks;

/**
 * Stops training when a monitored metric stops improving.
 *
 * <p>Monitors validation loss by default (lower is better). Set
 * {@code minimize = false} to monitor accuracy (higher is better).
 */
public class EarlyStopping implements TrainingCallback {

    private final int patience;
    private final boolean minimize;
    private double best;
    private int wait;
    private boolean stopped;

    /**
     * @param patience epochs with no improvement before stopping
     * @param minimize true if lower is better (loss), false if higher (accuracy)
     */
    public EarlyStopping(int patience, boolean minimize) {
        if (patience < 1) {
            throw new IllegalArgumentException("patience must be >= 1");
        }
        this.patience = patience;
        this.minimize = minimize;
        this.best = minimize ? Double.MAX_VALUE : -Double.MAX_VALUE;
        this.wait = 0;
        this.stopped = false;
    }

    /** Monitors validation loss with the given patience. */
    public EarlyStopping(int patience) {
        this(patience, true);
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
                stopped = true;
                return false;
            }
        }
        return true;
    }

    /** Returns the best monitored value seen so far. */
    public double bestValue() {
        return best;
    }

    /** Returns true if training was stopped early. */
    public boolean wasStopped() {
        return stopped;
    }

    /** Resets the callback for a new training run. */
    public void reset() {
        best = minimize ? Double.MAX_VALUE : -Double.MAX_VALUE;
        wait = 0;
        stopped = false;
    }
}
