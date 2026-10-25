package ml.metrics;

import ml.Vector;

/** Evaluation metrics for the classical ML models. */
public final class Metrics {

    private Metrics() {
    }

    /** Fraction of exactly matching predictions. */
    public static double accuracy(Vector yTrue, double[] yPred) {
        checkLengths(yTrue, yPred, "accuracy");
        int correct = 0;
        for (int i = 0; i < yPred.length; i++) {
            if (Double.compare(yTrue.get(i), yPred[i]) == 0) {
                correct++;
            }
        }
        return (double) correct / yPred.length;
    }

    /** Mean squared error (lower is better). */
    public static double meanSquaredError(Vector yTrue, double[] yPred) {
        checkLengths(yTrue, yPred, "meanSquaredError");
        double sum = 0.0;
        for (int i = 0; i < yPred.length; i++) {
            double diff = yTrue.get(i) - yPred[i];
            sum += diff * diff;
        }
        return sum / yPred.length;
    }

    /** Coefficient of determination (1 is perfect, 0 predicts the mean). */
    public static double r2Score(Vector yTrue, double[] yPred) {
        checkLengths(yTrue, yPred, "r2Score");
        double mean = yTrue.mean();
        double residual = 0.0;
        double total = 0.0;
        for (int i = 0; i < yPred.length; i++) {
            double diffPrediction = yTrue.get(i) - yPred[i];
            double diffMean = yTrue.get(i) - mean;
            residual += diffPrediction * diffPrediction;
            total += diffMean * diffMean;
        }
        if (total == 0.0) {
            return residual == 0.0 ? 1.0 : 0.0;
        }
        return 1.0 - residual / total;
    }

    private static void checkLengths(Vector yTrue, double[] yPred, String metric) {
        if (yTrue == null || yPred == null) {
            throw new IllegalArgumentException(metric + " requires non-null inputs");
        }
        if (yTrue.size() != yPred.length) {
            throw new IllegalArgumentException(metric + " requires aligned inputs: "
                    + yTrue.size() + " vs " + yPred.length);
        }
        if (yPred.length == 0) {
            throw new IllegalArgumentException(metric + " requires at least one sample");
        }
    }
}
