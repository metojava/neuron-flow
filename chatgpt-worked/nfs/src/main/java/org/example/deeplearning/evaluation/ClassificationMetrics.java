package org.example.deeplearning.evaluation;

/** Utilities for evaluating mutually exclusive class predictions. */
public final class ClassificationMetrics {
    private ClassificationMetrics() { }
    public static int argmax(double[] scores) {
        if (scores == null || scores.length == 0) throw new IllegalArgumentException("Scores must not be empty");
        int best = 0;
        for (int i = 0; i < scores.length; i++) {
            if (!Double.isFinite(scores[i])) throw new IllegalArgumentException("Scores must be finite");
            if (scores[i] > scores[best]) best = i;
        }
        return best;
    }
    /** Rows are actual classes; columns are predicted classes. */
    public static int[][] confusionMatrix(int[] actual, int[] predicted, int classes) {
        if (actual == null || predicted == null || actual.length == 0 || actual.length != predicted.length || classes <= 0)
            throw new IllegalArgumentException("Invalid labels or class count");
        int[][] result = new int[classes][classes];
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] < 0 || actual[i] >= classes || predicted[i] < 0 || predicted[i] >= classes)
                throw new IllegalArgumentException("Label outside class range");
            result[actual[i]][predicted[i]]++;
        }
        return result;
    }
    public static double accuracy(int[] actual, int[] predicted) {
        if (actual == null || predicted == null || actual.length == 0 || actual.length != predicted.length)
            throw new IllegalArgumentException("Labels must have equal nonzero lengths");
        int correct = 0;
        for (int i = 0; i < actual.length; i++) if (actual[i] == predicted[i]) correct++;
        return (double) correct / actual.length;
    }
}
