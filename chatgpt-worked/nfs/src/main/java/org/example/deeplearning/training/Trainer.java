package org.example.deeplearning.training;

import java.util.Random;

public class Trainer {
    private final int epochs;
    private final int batchSize;
    private final boolean shuffle;
    private final Random random;

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

    public int getEpochs() {
        return epochs;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public boolean shouldShuffle() {
        return shuffle;
    }

    public Random getRandom() {
        return random;
    }
}