package ml.model;

import ml.DataFrame;
import ml.Vector;

import java.util.Arrays;

/** Shared validation and numeric helpers for the built-in estimators. */
public final class ModelUtils {

    private ModelUtils() {
    }

    /**
     * Validates that a frame is model-ready and copies it into a row-major
     * matrix. Rejects categorical columns, non-finite values and empty data.
     */
    public static double[][] numericMatrix(DataFrame X) {
        if (X.rows() == 0 || X.columns() == 0) {
            throw new IllegalArgumentException("Feature frame must be non-empty");
        }
        double[][] matrix = X.toDoubleMatrix();
        for (int r = 0; r < matrix.length; r++) {
            for (int c = 0; c < matrix[r].length; c++) {
                if (!Double.isFinite(matrix[r][c])) {
                    throw new IllegalArgumentException("Feature (" + r + ", " + X.name(c)
                            + ") is not finite; impute missing values first");
                }
            }
        }
        return matrix;
    }

    /** Like {@link #numericMatrix(DataFrame)} but additionally checks a fitted feature count. */
    public static double[][] predictionMatrix(DataFrame X, int expectedFeatures) {
        double[][] matrix = numericMatrix(X);
        if (matrix[0].length != expectedFeatures) {
            throw new IllegalArgumentException("Model expects " + expectedFeatures
                    + " features but X has " + matrix[0].length);
        }
        return matrix;
    }

    /** Prediction-time validation that also guards against column reordering. */
    public static double[][] predictionMatrix(DataFrame X, String[] featureNames) {
        double[][] matrix = predictionMatrix(X, featureNames.length);
        for (int c = 0; c < featureNames.length; c++) {
            if (!featureNames[c].equals(X.name(c))) {
                throw new IllegalArgumentException("Feature " + c + " is '" + X.name(c)
                        + "' but the model was trained on '" + featureNames[c] + "'");
            }
        }
        return matrix;
    }

    /** Fails with a descriptive message when a model is used before fitting. */
    public static void requireFitted(boolean fitted, String model) {
        if (!fitted) {
            throw new IllegalStateException(model + " has not been fitted");
        }
    }

    /** Validates target length and finiteness, then returns a defensive copy. */
    public static double[] targetVector(Vector y, int rows) {
        if (y.size() != rows) {
            throw new IllegalArgumentException("Target has " + y.size() + " values but X has " + rows + " rows");
        }
        double[] target = y.toArray();
        for (int i = 0; i < target.length; i++) {
            if (!Double.isFinite(target[i])) {
                throw new IllegalArgumentException("Target contains a non-finite value at index " + i);
            }
        }
        return target;
    }

    /** Distinct target values in ascending order. */
    public static double[] classesOf(Vector y) {
        return Arrays.stream(y.toArray()).distinct().sorted().toArray();
    }

    /** Numerically stable logistic sigmoid. */
    public static double sigmoid(double z) {
        if (z >= 0.0) {
            double e = Math.exp(-z);
            return 1.0 / (1.0 + e);
        }
        double e = Math.exp(z);
        return e / (1.0 + e);
    }

    /** Index of the largest value; {@code 0} for an empty array. */
    public static int argmax(double[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    /** Index of the class value nearest to {@code value}. */
    public static int nearestClass(double[] classes, double value) {
        int best = 0;
        double bestDistance = Double.POSITIVE_INFINITY;
        for (int i = 0; i < classes.length; i++) {
            double distance = Math.abs(classes[i] - value);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = i;
            }
        }
        return best;
    }

    /** Squared Euclidean distance between two rows. */
    public static double squaredDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
