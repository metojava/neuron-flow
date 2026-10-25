package util;

/**
 * Computes class weights for imbalanced datasets.
 * Supports inverse frequency and balanced modes.
 */
public final class ClassWeights {

    private ClassWeights() {
    }

    /**
     * Computes inverse-frequency weights: weight[c] = total / (numClasses * count[c]).
     *
     * @param labels     array of class labels
     * @param numClasses number of classes
     * @return weight per class
     */
    public static float[] inverseFrequency(int[] labels, int numClasses) {
        if (labels == null || labels.length == 0) {
            throw new IllegalArgumentException("labels must not be empty");
        }
        int[] counts = new int[numClasses];
        for (int label : labels) {
            if (label >= 0 && label < numClasses) {
                counts[label]++;
            }
        }
        float[] weights = new float[numClasses];
        for (int c = 0; c < numClasses; c++) {
            if (counts[c] > 0) {
                weights[c] = (float) labels.length / (numClasses * counts[c]);
            } else {
                weights[c] = 1.0f;
            }
        }
        return weights;
    }

    /**
     * Computes balanced weights: same as inverse frequency but
     * normalized so the mean weight is 1.
     *
     * @param labels     array of class labels
     * @param numClasses number of classes
     * @return weight per class (mean = 1)
     */
    public static float[] balanced(int[] labels, int numClasses) {
        float[] weights = inverseFrequency(labels, numClasses);
        double mean = 0.0;
        for (float w : weights) mean += w;
        mean /= numClasses;
        for (int i = 0; i < weights.length; i++) {
            weights[i] /= (float) mean;
        }
        return weights;
    }
}
