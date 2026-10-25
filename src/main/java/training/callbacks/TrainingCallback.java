package training.callbacks;

/**
 * Hook into the training loop. Implementations can log metrics,
 * save checkpoints, or stop training early.
 */
public interface TrainingCallback {

    /**
     * Called at the end of each epoch.
     *
     * @param epoch      zero-based epoch index
     * @param trainLoss  mean training loss for the epoch
     * @param valLoss    validation loss, or NaN if not computed
     * @param valAcc     validation accuracy, or NaN if not computed
     * @return true to continue training, false to stop
     */
    boolean onEpochEnd(int epoch, double trainLoss,
                       double valLoss, double valAcc);
}
