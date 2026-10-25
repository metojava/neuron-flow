package evaluation;

/** Metrics for one-dimensional regression predictions. */
public final class RegressionMetrics {
    private RegressionMetrics() {
    }

    public static double meanAbsoluteError(float[] predicted, float[] actual) {
        validate(predicted, actual);
        double sum = 0.0;
        for (int i = 0; i < actual.length; i++) {
            sum += Math.abs((double) predicted[i] - actual[i]);
        }
        return sum / actual.length;
    }

    public static double meanSquaredError(float[] predicted, float[] actual) {
        validate(predicted, actual);
        double sum = 0.0;
        for (int i = 0; i < actual.length; i++) {
            double error = (double) predicted[i] - actual[i];
            sum += error * error;
        }
        return sum / actual.length;
    }

    public static double rootMeanSquaredError(float[] predicted, float[] actual) {
        return Math.sqrt(meanSquaredError(predicted, actual));
    }

    /** Returns 1 for a perfect constant target, otherwise 0 when variance is zero. */
    public static double rSquared(float[] predicted, float[] actual) {
        validate(predicted, actual);
        double mean = 0.0;
        for (float value : actual) mean += value;
        mean /= actual.length;
        double residual = 0.0;
        double total = 0.0;
        for (int i = 0; i < actual.length; i++) {
            double error = (double) actual[i] - predicted[i];
            double deviation = actual[i] - mean;
            residual += error * error;
            total += deviation * deviation;
        }
        return total == 0.0 ? (residual == 0.0 ? 1.0 : 0.0)
            : 1.0 - residual / total;
    }

    private static void validate(float[] predicted, float[] actual) {
        if (predicted == null || actual == null || actual.length == 0
            || predicted.length != actual.length) {
            throw new IllegalArgumentException("Prediction and target sizes must match");
        }
        for (int i = 0; i < actual.length; i++) {
            if (!Float.isFinite(predicted[i]) || !Float.isFinite(actual[i])) {
                throw new IllegalArgumentException("Values must be finite");
            }
        }
    }
}
