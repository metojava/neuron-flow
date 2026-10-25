package ml.model;

import ml.DataFrame;
import ml.Estimator;
import ml.Predictor;
import ml.Vector;
import ml.metrics.Metrics;

import java.util.Random;

/**
 * Logistic regression trained with shuffled mini-batch stochastic gradient
 * descent and optional L2 regularisation.
 *
 * <p>Binary problems use a single sigmoid model. Problems with more than two
 * classes are handled one-vs-rest: one sigmoid model per class, with the
 * per-class probabilities normalised at prediction time. Class labels may be
 * arbitrary finite doubles; they are ordered ascending, so for binary data the
 * larger label is treated as the positive class.
 */
public final class LogisticRegression implements Estimator<LogisticRegression>, Predictor {

    /** Default number of samples per gradient update. */
    public static final int DEFAULT_BATCH_SIZE = 32;

    private final double learningRate;
    private final int maxEpochs;
    private final double l2;
    private final double tolerance;
    private final int batchSize;
    private final long seed;
    private final boolean fitIntercept;

    private double[] classes;
    private double[][] weights;
    private double[] intercepts;
    private int[] epochsRun;
    private String[] featureNames;
    private boolean fitted;

    public LogisticRegression() {
        this(0.1, 1_000, 0.0, 1e-8, DEFAULT_BATCH_SIZE, 42L, true);
    }

    public LogisticRegression(double learningRate, int maxEpochs, double l2,
                              double tolerance, boolean fitIntercept) {
        this(learningRate, maxEpochs, l2, tolerance, DEFAULT_BATCH_SIZE, 42L, fitIntercept);
    }

    /**
     * @param learningRate  gradient-descent step size, {@code > 0}
     * @param maxEpochs     maximum number of passes over the data
     * @param l2            L2 penalty strength, {@code >= 0}
     * @param tolerance     stop when the epoch log-loss changes by less than this
     * @param batchSize     samples per gradient update; {@code 1} is pure SGD
     * @param seed          seed for the per-epoch shuffle
     * @param fitIntercept  whether to learn a bias term
     */
    public LogisticRegression(double learningRate, int maxEpochs, double l2, double tolerance,
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
    public LogisticRegression fit(DataFrame X, Vector y) {
        double[][] matrix = ModelUtils.numericMatrix(X);
        double[] target = ModelUtils.targetVector(y, matrix.length);
        double[] distinct = ModelUtils.classesOf(y);
        if (distinct.length < 2) {
            throw new IllegalArgumentException("LogisticRegression needs at least two classes");
        }

        double[][] newWeights = new double[distinct.length == 2 ? 1 : distinct.length][];
        double[] newIntercepts = new double[newWeights.length];
        int[] newEpochs = new int[newWeights.length];
        for (int k = 0; k < newWeights.length; k++) {
            double[] binaryTarget = new double[target.length];
            if (distinct.length == 2) {
                for (int i = 0; i < target.length; i++) {
                    binaryTarget[i] = target[i] == distinct[1] ? 1.0 : 0.0;
                }
            } else {
                for (int i = 0; i < target.length; i++) {
                    binaryTarget[i] = target[i] == distinct[k] ? 1.0 : 0.0;
                }
            }
            BinaryFit fit = trainBinary(matrix, binaryTarget, seed + k);
            newWeights[k] = fit.weights;
            newIntercepts[k] = fit.intercept;
            newEpochs[k] = fit.epochs;
        }

        classes = distinct;
        weights = newWeights;
        intercepts = newIntercepts;
        epochsRun = newEpochs;
        featureNames = X.names();
        fitted = true;
        return this;
    }

    private BinaryFit trainBinary(double[][] matrix, double[] target, long modelSeed) {
        int rows = matrix.length;
        int features = matrix[0].length;
        double[] newWeights = new double[features];
        double newIntercept = 0.0;
        int[] order = new int[rows];
        for (int i = 0; i < rows; i++) {
            order[i] = i;
        }
        Random random = new Random(modelSeed);
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
                    double score = newIntercept;
                    for (int c = 0; c < features; c++) {
                        score += newWeights[c] * matrix[r][c];
                    }
                    double probability = ModelUtils.sigmoid(score);
                    double error = probability - target[r];
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

            double loss = logLoss(matrix, target, newWeights, newIntercept);
            completedEpochs = epoch + 1;
            if (epoch > 0 && Math.abs(previousLoss - loss) <= tolerance) {
                break;
            }
            previousLoss = loss;
        }
        return new BinaryFit(newWeights, newIntercept, completedEpochs);
    }

    private double logLoss(double[][] matrix, double[] target, double[] currentWeights, double currentIntercept) {
        double loss = 0.0;
        for (int r = 0; r < matrix.length; r++) {
            double score = currentIntercept;
            for (int c = 0; c < currentWeights.length; c++) {
                score += currentWeights[c] * matrix[r][c];
            }
            double probability = ModelUtils.sigmoid(score);
            double clipped = Math.min(1.0 - 1e-15, Math.max(1e-15, probability));
            loss -= target[r] * Math.log(clipped) + (1.0 - target[r]) * Math.log(1.0 - clipped);
        }
        double penalty = 0.0;
        for (double weight : currentWeights) {
            penalty += weight * weight;
        }
        return loss / matrix.length + 0.5 * l2 * penalty;
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
        ModelUtils.requireFitted(fitted, "LogisticRegression");
        double[][] decisions = decisionValues(X);
        double[] predictions = new double[decisions.length];
        for (int r = 0; r < decisions.length; r++) {
            int index = decisions[r].length == 1
                    ? (decisions[r][0] >= 0.0 ? 1 : 0)
                    : ModelUtils.argmax(decisions[r]);
            predictions[r] = classes[index];
        }
        return predictions;
    }

    @Override
    public double[][] predictProbabilities(DataFrame X) {
        ModelUtils.requireFitted(fitted, "LogisticRegression");
        double[][] decisions = decisionValues(X);
        double[][] result = new double[decisions.length][];
        if (decisions[0].length == 1) {
            for (int r = 0; r < decisions.length; r++) {
                double positive = ModelUtils.sigmoid(decisions[r][0]);
                result[r] = new double[]{1.0 - positive, positive};
            }
            return result;
        }
        for (int r = 0; r < decisions.length; r++) {
            double[] probabilities = new double[decisions[r].length];
            double sum = 0.0;
            for (int k = 0; k < probabilities.length; k++) {
                probabilities[k] = ModelUtils.sigmoid(decisions[r][k]);
                sum += probabilities[k];
            }
            if (sum <= 0.0) {
                for (int k = 0; k < probabilities.length; k++) {
                    probabilities[k] = 1.0 / probabilities.length;
                }
            } else {
                for (int k = 0; k < probabilities.length; k++) {
                    probabilities[k] /= sum;
                }
            }
            result[r] = probabilities;
        }
        return result;
    }

    private double[][] decisionValues(DataFrame X) {
        double[][] matrix = ModelUtils.predictionMatrix(X, featureNames);
        double[][] decisions = new double[matrix.length][weights.length];
        for (int r = 0; r < matrix.length; r++) {
            for (int k = 0; k < weights.length; k++) {
                double score = intercepts[k];
                for (int c = 0; c < weights[k].length; c++) {
                    score += weights[k][c] * matrix[r][c];
                }
                decisions[r][k] = score;
            }
        }
        return decisions;
    }

    /** Accuracy on the supplied data. */
    public double score(DataFrame X, Vector y) {
        return Metrics.accuracy(y, predict(X));
    }

    /** Class labels in ascending order; probability columns follow this order. */
    public double[] classes() {
        ModelUtils.requireFitted(fitted, "LogisticRegression");
        return classes.clone();
    }

    /**
     * Coefficients with shape {@code [models][features]}. Binary models have a
     * single row; multiclass models have one row per class.
     */
    public double[][] coefficients() {
        ModelUtils.requireFitted(fitted, "LogisticRegression");
        double[][] copy = new double[weights.length][];
        for (int k = 0; k < weights.length; k++) {
            copy[k] = weights[k].clone();
        }
        return copy;
    }

    public double[] intercepts() {
        ModelUtils.requireFitted(fitted, "LogisticRegression");
        return intercepts.clone();
    }

    public int[] epochsRun() {
        ModelUtils.requireFitted(fitted, "LogisticRegression");
        return epochsRun.clone();
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

    private record BinaryFit(double[] weights, double intercept, int epochs) {
    }
}
