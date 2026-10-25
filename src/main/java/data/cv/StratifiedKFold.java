package data.cv;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Stratified k-fold cross-validation. Preserves class proportions
 * in each fold.
 */
public final class StratifiedKFold {

    private final int nFolds;
    private final int nSamples;
    private final int[] foldId;

    /**
     * @param labels array of class labels (int) for each sample
     * @param nFolds number of folds
     * @param seed   RNG seed for shuffling
     */
    public StratifiedKFold(int[] labels, int nFolds, long seed) {
        if (labels == null || labels.length < nFolds || nFolds < 2) {
            throw new IllegalArgumentException(
                "Need at least nFolds >= 2 samples");
        }
        this.nSamples = labels.length;
        this.nFolds = nFolds;
        this.foldId = new int[nSamples];

        // Group indices by class
        int maxClass = 0;
        for (int l : labels) {
            if (l > maxClass) maxClass = l;
        }
        int numClasses = maxClass + 1;

        List<List<Integer>> classIndices = new ArrayList<>();
        for (int c = 0; c < numClasses; c++) {
            classIndices.add(new ArrayList<>());
        }
        for (int i = 0; i < nSamples; i++) {
            classIndices.get(labels[i]).add(i);
        }

        Random rng = new Random(seed);
        for (List<Integer> indices : classIndices) {
            // Shuffle within class
            for (int i = indices.size() - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = indices.get(i);
                indices.set(i, indices.get(j));
                indices.set(j, tmp);
            }
            // Assign folds round-robin
            for (int i = 0; i < indices.size(); i++) {
                foldId[indices.get(i)] = i % nFolds;
            }
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
