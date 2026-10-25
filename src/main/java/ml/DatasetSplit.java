package ml;

import java.util.Objects;
import java.util.Random;

/**
 * Deterministic train/test splitting for frames and target vectors.
 */
public final class DatasetSplit {

    private DatasetSplit() {
    }

    /** Result of a split; the four components are aligned by row. */
    public record Split(DataFrame train, Vector yTrain, DataFrame test, Vector yTest) {
        public Split {
            Objects.requireNonNull(train, "train");
            Objects.requireNonNull(yTrain, "yTrain");
            Objects.requireNonNull(test, "test");
            Objects.requireNonNull(yTest, "yTest");
        }

        public int trainSize() {
            return train.rows();
        }

        public int testSize() {
            return test.rows();
        }
    }

    /**
     * Shuffles and splits rows.
     *
     * @param testFraction fraction of rows placed in the test set, in {@code (0, 1)}
     * @param seed random seed for reproducible shuffling
     */
    public static Split split(DataFrame X, Vector y, double testFraction, long seed) {
        Objects.requireNonNull(X, "X");
        Objects.requireNonNull(y, "y");
        if (X.rows() != y.size()) {
            throw new IllegalArgumentException("X has " + X.rows() + " rows but y has " + y.size() + " values");
        }
        if (X.rows() < 2) {
            throw new IllegalArgumentException("At least two rows are required to split a dataset");
        }
        if (!(testFraction > 0.0 && testFraction < 1.0)) {
            throw new IllegalArgumentException("testFraction must lie in (0, 1)");
        }

        int n = X.rows();
        int testSize = (int) Math.round(n * testFraction);
        testSize = Math.max(1, Math.min(n - 1, testSize));

        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        Random random = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        int[] trainIndices = new int[n - testSize];
        int[] testIndices = new int[testSize];
        System.arraycopy(indices, 0, testIndices, 0, testSize);
        System.arraycopy(indices, testSize, trainIndices, 0, n - testSize);

        return new Split(X.takeRows(trainIndices), y.take(trainIndices),
                X.takeRows(testIndices), y.take(testIndices));
    }
}
