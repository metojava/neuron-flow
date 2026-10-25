package data;

import java.util.Random;

/** Reproducible paired training and test split for float feature and target rows. */
public final class DatasetSplit {
    private DatasetSplit() {
    }

    /**
     * Splits paired rows after shuffling with {@code seed}. Both partitions are
     * nonempty. Returned rows are copies, so changes do not alter the originals.
     */
    public static Split split(float[][] features, float[][] targets,
                              double testFraction, long seed) {
        if (features == null || targets == null || features.length < 2
            || features.length != targets.length
            || !Double.isFinite(testFraction)
            || testFraction <= 0.0 || testFraction >= 1.0) {
            throw new IllegalArgumentException("Invalid dataset or test fraction");
        }
        validateRows(features);
        validateRows(targets);
        int n = features.length;
        int testCount = Math.max(1, Math.min(n - 1, (int) Math.round(n * testFraction)));
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Random random = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        int trainCount = n - testCount;
        float[][] trainFeatures = new float[trainCount][];
        float[][] trainTargets = new float[trainCount][];
        float[][] testFeatures = new float[testCount][];
        float[][] testTargets = new float[testCount][];
        for (int i = 0; i < n; i++) {
            int source = order[i];
            if (i < trainCount) {
                trainFeatures[i] = features[source].clone();
                trainTargets[i] = targets[source].clone();
            } else {
                int destination = i - trainCount;
                testFeatures[destination] = features[source].clone();
                testTargets[destination] = targets[source].clone();
            }
        }
        return new Split(trainFeatures, trainTargets, testFeatures, testTargets);
    }

    private static void validateRows(float[][] rows) {
        int width = rows[0] == null ? 0 : rows[0].length;
        if (width == 0) throw new IllegalArgumentException("Rows must be nonempty");
        for (float[] row : rows) {
            if (row == null || row.length != width) {
                throw new IllegalArgumentException("Rows must have equal dimensions");
            }
            for (float value : row) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("Data must be finite");
                }
            }
        }
    }

    public record Split(float[][] trainFeatures, float[][] trainTargets,
                        float[][] testFeatures, float[][] testTargets) {
    }
}
