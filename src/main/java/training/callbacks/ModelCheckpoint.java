package training.callbacks;

import layers.NeuralNetwork;
import layers.ModelFormat;

import java.nio.file.Path;

/**
 * Saves the model to disk whenever the monitored metric improves.
 *
 * <p>Monitors validation loss by default (lower is better). Set
 * {@code minimize = false} to monitor accuracy (higher is better).
 */
public class ModelCheckpoint implements TrainingCallback {

    private final Path path;
    private final ModelFormat format;
    private final boolean minimize;
    private double best;
    private boolean saved;

    /**
     * @param path      destination file path
     * @param format    file format (BINARY, JSON, CSV)
     * @param minimize  true if lower is better (loss), false if higher (accuracy)
     */
    public ModelCheckpoint(Path path, ModelFormat format, boolean minimize) {
        this.path = path;
        this.format = format;
        this.minimize = minimize;
        this.best = minimize ? Double.MAX_VALUE : -Double.MAX_VALUE;
        this.saved = false;
    }

    /** Monitors validation loss and saves in BINARY format. */
    public ModelCheckpoint(Path path) {
        this(path, ModelFormat.BINARY, true);
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
            saved = true;
        }
        return true;
    }

    /** Returns the best monitored value seen so far. */
    public double bestValue() {
        return best;
    }

    /** Returns true if at least one checkpoint was saved. */
    public boolean wasSaved() {
        return saved;
    }
}
