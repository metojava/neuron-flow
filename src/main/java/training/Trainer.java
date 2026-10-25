package training;

import java.util.Random;

/**
 * Training configuration: how many epochs, how large the
 * mini-batches, whether to reshuffle each epoch, and the seed
 * for reproducible shuffling. Read by
 * {@link layers.NeuralNetwork#fit}.
 */
public class Trainer {

    private final int epochs;
    private final int batchSize;
    private final boolean shuffle;
    private final Random random;

    /**
     * Builds a training configuration.
     *
     * @param epochs     full passes over the data; must be &gt; 0
     * @param batchSize  samples per gradient update; must be &gt; 0
     * @param shuffle    reshuffle row order every epoch
     * @param seed       seed for the shuffle RNG
     * @throws IllegalArgumentException if epochs or batch size
     *                                  is not positive
     */
    public Trainer(int epochs, int batchSize, boolean shuffle,
                   long seed) {
        if (epochs <= 0 || batchSize <= 0) {
            throw new IllegalArgumentException(
                "Epochs and batch size must be positive");
        }

        this.epochs = epochs;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.random = new Random(seed);
    }

    /** @return number of full passes over the data */
    public int getEpochs() {
        return epochs;
    }

    /** @return samples per gradient update */
    public int getBatchSize() {
        return batchSize;
    }

    /** @return true if rows are shuffled each epoch */
    public boolean shouldShuffle() {
        return shuffle;
    }

    /** @return the RNG used for shuffling */
    public Random getRandom() {
        return random;
    }
}
