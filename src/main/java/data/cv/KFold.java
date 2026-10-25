package data.cv;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * K-fold cross-validation splitter. Produces train/test index arrays
 * for each fold.
 */
public final class KFold {

    private final int nFolds;
    private final int nSamples;
    private final int[] foldId;

    /**
     * @param nSamples number of samples
     * @param nFolds   number of folds
     * @param seed     RNG seed for shuffling
     */
    public KFold(int nSamples, int nFolds, long seed) {
        if (nSamples < nFolds || nFolds < 2) {
            throw new IllegalArgumentException(
                "nSamples must be >= nFolds >= 2");
        }
        this.nSamples = nSamples;
        this.nFolds = nFolds;
        this.foldId = new int[nSamples];

        // Shuffle indices
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) indices[i] = i;
        Random rng = new Random(seed);
        for (int i = nSamples - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        // Assign fold ids
        for (int i = 0; i < nSamples; i++) {
            foldId[indices[i]] = i % nFolds;
        }
    }

    /** Number of folds. */
    public int nFolds() {
        return nFolds;
    }

    /**
     * Returns train and test indices for the given fold.
     *
     * @param fold zero-based fold index
     * @return int[2][] where [0] = train indices, [1] = test indices
     */
    public int[][] split(int fold) {
        if (fold < 0 || fold >= nFolds) {
            throw new IllegalArgumentException("fold out of range");
        }
        List<Integer> train = new ArrayList<>();
        List<Integer> test = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            if (foldId[i] == fold) {
                test.add(i);
            } else {
                train.add(i);
            }
        }
        int[] trainArr = new int[train.size()];
        int[] testArr = new int[test.size()];
        for (int i = 0; i < train.size(); i++) trainArr[i] = train.get(i);
        for (int i = 0; i < test.size(); i++) testArr[i] = test.get(i);
        return new int[][] { trainArr, testArr };
    }

}
