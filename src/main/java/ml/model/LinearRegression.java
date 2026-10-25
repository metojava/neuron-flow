package ml.model;

import ml.DataFrame;
import ml.Estimator;
import ml.Predictor;
import ml.Vector;
import ml.metrics.Metrics;

import java.util.Random;

/**
 * Ordinary least-squares linear regression trained with shuffled mini-batch
 * stochastic gradient descent and optional L2 regularisation.
 *
 * <p>Gradient descent is scale sensitive: standardise features with a
 * {@link ml.preprocessing.StandardScaler} before fitting when they have very
 * different magnitudes.
 */
public final class LinearRegression implements Estimator<LinearRegression>, Predictor {

    /** Default number of samples per gradient update. */
    public static final int DEFAULT_BATCH_SIZE = 32;

    private final double learningRate;
    private final int maxEpochs;
    private final double l2;
    private final double tolerance;
    private final int batchSize;
    private final long seed;
    private final boolean fitIntercept;

    private double[] weights;
    private double intercept;
    private String[] featureNames;
    private boolean fitted;
    private int epochsRun;

    public LinearRegression() {
        this(0.05, 2_000, 0.0, 1e-10, DEFAULT_BATCH_SIZE, 42L, true);
    }

    /**
     * @param learningRate  gradient-descent step size, {@code > 0}
     * @param maxEpochs     maximum number of passes over the data
     * @param l2            L2 penalty strength, {@code >= 0}
     * @param tolerance     stop when the epoch loss changes by less than this
     * @param fitIntercept  whether to learn a bias term
     */
    public LinearRegression(double learningRate, int maxEpochs, double l2,
                            double tolerance, boolean fitIntercept) {
        this(learningRate, maxEpochs, l2, tolerance, DEFAULT_BATCH_SIZE, 42L, fitIntercept);
    }

    /**
     * @param learningRate  gradient-descent step size, {@code > 0}
     * @param maxEpochs     maximum number of passes over the data
     * @param l2            L2 penalty strength, {@code >= 0}
     * @param tolerance     stop when the epoch loss changes by less than this
     * @param batchSize     samples per gradient update; {@code 1} is pure SGD
     * @param seed          seed for the per-epoch shuffle
     * @param fitIntercept  whether to learn a bias term
     */
    public LinearRegression(double learningRate, int maxEpochs, double l2, double tolerance,
                            int batchSize, long seed, boolean fitIntercept) {
        if (!Double.isFinite(learningRate) || learningRate <= 0.0) {
            throw new IllegalArgumentException("learningRate must be positive and finite");
        }
        if (maxEpochs <= 0) {
            throw new IllegalArgumentException("maxEpochs must be positive");
        }
        if (!Double.isFinite(l2) || l2 < 0.0) {
            throw new IllegalArgumentException("l2 must be finite and non-negative");
        }
        if (!Double.isFinite(tolerance) || tolerance < 0.0) {
            throw new IllegalArgumentException("tolerance must be finite and non-negative");
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        this.learningRate = learningRate;
        this.maxEpochs = maxEpochs;
        this.l2 = l2;
        this.tolerance = tolerance;
        this.batchSize = batchSize;
        this.seed = seed;
        this.fitIntercept = fitIntercept;
    }

    @Override
    public LinearRegression fit(DataFrame X, Vector y) {
        double[][] matrix = ModelUtils.numericMatrix(X);
        double[] target = ModelUtils.targetVector(y, matrix.length);
        int rows = matrix.length;
        int features = matrix[0].length;

        double[] newWeights = new double[features];
        double newIntercept = 0.0;
        if (fitIntercept) {
            double sum = 0.0;
            for (double value : target) {
                sum += value;
            }
            newIntercept = sum / rows;
        }

        int[] order = new int[rows];
        for (int i = 0; i < rows; i++) {
            order[i] = i;
        }
        Random random = new Random(seed);

        double previousLoss = Double.POSITIVE_INFINITY;
        int completedEpochs = 0;
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            shuffle(order, random);
            for (int start = 0; start < rows; start += batchSize) {
                int end = Math.min(rows, start + batchSize);
                int batchRows = end - start;
                double[] gradient = new double[features];
                double gradientIntercept = 0.0;
                for (int position = start; position < end; position++) {
                    int r = order[position];
                    double prediction = newIntercept;
                    for (int c = 0; c < features; c++) {
                        prediction += newWeights[c] * matrix[r][c];
                    }
                    double error = prediction - target[r];
                    for (int c = 0; c < features; c++) {
                        gradient[c] += error * matrix[r][c];
                    }
                    gradientIntercept += error;
                }

                double inverseBatch = 1.0 / batchRows;
                for (int c = 0; c < features; c++) {
                    gradient[c] = gradient[c] * inverseBatch + l2 * newWeights[c];
                    newWeights[c] -= learningRate * gradient[c];
                }
                if (fitIntercept) {
                    newIntercept -= learningRate * gradientIntercept * inverseBatch;
                }
            }

            double loss = loss(matrix, target, newWeights, newIntercept);
            completedEpochs = epoch + 1;
            if (epoch > 0 && Math.abs(previousLoss - loss) <= tolerance) {
                break;
            }
            previousLoss = loss;
        }

        weights = newWeights;
        intercept = newIntercept;
        featureNames = X.names();
        epochsRun = completedEpochs;
        fitted = true;
        return this;
    }

    private double loss(double[][] matrix, double[] target, double[] currentWeights, double currentIntercept) {
        double squaredError = 0.0;
        for (int r = 0; r < matrix.length; r++) {
            double prediction = currentIntercept;
            for (int c = 0; c < currentWeights.length; c++) {
                prediction += currentWeights[c] * matrix[r][c];
            }
            double error = prediction - target[r];
            squaredError += error * error;
        }
        double penalty = 0.0;
        for (double weight : currentWeights) {
            penalty += weight * weight;
        }
        return squaredError / matrix.length + 0.5 * l2 * penalty;
    }

    private static void shuffle(int[] order, Random random) {
        for (int i = order.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
    }

    @Override
    public double[] predict(DataFrame X) {
        ModelUtils.requireFitted(fitted, "LinearRegression");
        double[][] matrix = ModelUtils.predictionMatrix(X, featureNames);
        double[] predictions = new double[matrix.length];
        for (int r = 0; r < matrix.length; r++) {
            double value = intercept;
            for (int c = 0; c < weights.length; c++) {
                value += weights[c] * matrix[r][c];
            }
            predictions[r] = value;
        }
        return predictions;
    }

    @Override
    public double[][] predictProbabilities(DataFrame X) {
        double[] predictions = predict(X);
        double[][] result = new double[predictions.length][1];
        for (int i = 0; i < predictions.length; i++) {
            result[i][0] = predictions[i];
        }
        return result;
    }

    /** R-squared on the supplied data. */
    public double score(DataFrame X, Vector y) {
        return Metrics.r2Score(y, predict(X));
    }

    public double[] coefficients() {
        ModelUtils.requireFitted(fitted, "LinearRegression");
        return weights.clone();
    }

    public double intercept() {
        ModelUtils.requireFitted(fitted, "LinearRegression");
        return intercept;
    }

    public int epochsRun() {
        ModelUtils.requireFitted(fitted, "LinearRegression");
        return epochsRun;
    }

    public double learningRate() {
        return learningRate;
    }

    public int maxEpochs() {
        return maxEpochs;
    }

    public double l2() {
        return l2;
    }

    public double tolerance() {
        return tolerance;
    }

    public int batchSize() {
        return batchSize;
    }

    public long seed() {
        return seed;
    }

    public boolean fitIntercept() {
        return fitIntercept;
    }
}
