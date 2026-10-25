package optim;

/**
 * Learning-rate schedule that maps an epoch (or step) index to a
 * learning rate. Implementations are stateless: the schedule is a pure
 * function of the index and the initial learning rate.
 */
public interface LearningRateSchedule {

    /**
     * Returns the learning rate for the given epoch (0-based).
     *
     * @param epoch zero-based epoch index
     * @return the learning rate to use
     */
    float learningRate(int epoch);

    /**
     * Constant learning rate (no decay).
     *
     * @param lr the fixed learning rate
     * @return a constant schedule
     */
    static LearningRateSchedule constant(float lr) {
        return epoch -> lr;
    }

    /**
     * Step decay: multiply by {@code factor} every {@code stepSize}
     * epochs.
     *
     * @param initialLr starting learning rate
     * @param factor    multiplier applied at each step (e.g. 0.1)
     * @param stepSize  epochs between steps
     * @return a step-decay schedule
     */
    static LearningRateSchedule stepDecay(float initialLr,
                                           float factor, int stepSize) {
        if (stepSize <= 0) {
            throw new IllegalArgumentException("stepSize must be positive");
        }
        return epoch -> initialLr * (float) Math.pow(
            factor, epoch / stepSize);
    }

    /**
     * Exponential decay: lr = initialLr * exp(-decay * epoch).
     *
     * @param initialLr starting learning rate
     * @param decay     decay rate per epoch
     * @return an exponential-decay schedule
     */
    static LearningRateSchedule exponential(float initialLr, float decay) {
        return epoch -> initialLr * (float) Math.exp(-decay * epoch);
    }

    /**
     * Cosine annealing: lr oscillates between 0 and initialLr following
     * a cosine curve over {@code totalEpochs} epochs.
     *
     * @param initialLr   starting (maximum) learning rate
     * @param totalEpochs number of epochs in one cycle
     * @return a cosine-annealing schedule
     */
    static LearningRateSchedule cosineAnnealing(float initialLr,
                                                int totalEpochs) {
        if (totalEpochs <= 0) {
            throw new IllegalArgumentException("totalEpochs must be positive");
        }
        return epoch -> {
            int e = Math.min(epoch, totalEpochs);
            return initialLr * 0.5f * (1.0f + (float) Math.cos(
                Math.PI * e / totalEpochs));
        };
    }

    /**
     * Linear warmup followed by cosine annealing. The learning rate
     * ramps from 0 to {@code initialLr} over {@code warmupEpochs}
     * epochs, then decays with cosine over the remaining epochs.
     *
     * @param initialLr     peak learning rate
     * @param warmupEpochs  number of warmup epochs
     * @param totalEpochs   total number of epochs
     * @return a warmup + cosine schedule
     */
    static LearningRateSchedule warmupCosine(float initialLr,
                                             int warmupEpochs,
                                             int totalEpochs) {
        if (warmupEpochs < 0 || totalEpochs <= 0
            || warmupEpochs >= totalEpochs) {
            throw new IllegalArgumentException("invalid warmup/total epochs");
        }
        return epoch -> {
            if (epoch < warmupEpochs) {
                return initialLr * (epoch + 1.0f) / warmupEpochs;
            }
            int remaining = totalEpochs - warmupEpochs;
            int e = epoch - warmupEpochs;
            return initialLr * 0.5f * (1.0f + (float) Math.cos(
                Math.PI * e / remaining));
        };
    }
}
