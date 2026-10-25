package ml.model;

import ml.DataFrame;
import ml.Estimator;
import ml.Predictor;
import ml.Vector;
import ml.kernel.Kernel;
import ml.metrics.Metrics;

import java.util.ArrayList;
import java.util.List;

/**
 * Kernel support-vector machine trained with sequential minimal optimisation
 * (SMO). Binary problems use a single decision function; multiclass problems
 * are handled one-vs-rest.
 *
 * <p>The kernel matrix is precomputed once during fitting, which costs
 * {@code O(n^2)} memory. This is appropriate for the small and medium-sized
 * datasets targeted by this library; for very large problems prefer a primal
 * linear solver or an approximate kernel expansion.
 *
 * <p>Probability estimates are obtained by applying a sigmoid to the decision
 * values and normalising; they indicate confidence but are not calibrated.
 */
public final class SVM implements Estimator<SVM>, Predictor {

    private final double c;
    private final double tolerance;
    private final int maxPasses;
    private final Kernel kernel;

    private double[] classes;
    private double[][] trainingData;
    private int[][] supportIndices;
    private double[][] supportCoefficients;
    private double[] biases;
    private String[] featureNames;
    private boolean fitted;

    public SVM() {
        this(1.0, Kernel.linear(), 1e-3, 20);
    }

    public SVM(Kernel kernel) {
        this(1.0, kernel, 1e-3, 20);
    }

    public SVM(double c, Kernel kernel) {
        this(c, kernel, 1e-3, 20);
    }

    /**
     * @param c         regularisation parameter; larger values fit the training data more closely
     * @param kernel    Mercer kernel
     * @param tolerance KKT violation threshold
     * @param maxPasses maximum number of SMO passes without a change
     */
    public SVM(double c, Kernel kernel, double tolerance, int maxPasses) {
        if (!Double.isFinite(c) || c <= 0.0) {
            throw new IllegalArgumentException("c must be positive and finite");
        }
        if (kernel == null) {
            throw new IllegalArgumentException("kernel must not be null");
        }
        if (!Double.isFinite(tolerance) || tolerance <= 0.0) {
            throw new IllegalArgumentException("tolerance must be positive and finite");
        }
        if (maxPasses <= 0) {
            throw new IllegalArgumentException("maxPasses must be positive");
        }
        this.c = c;
        this.tolerance = tolerance;
        this.maxPasses = maxPasses;
        this.kernel = kernel;
    }

    @Override
    public SVM fit(DataFrame X, Vector y) {
        double[][] matrix = ModelUtils.numericMatrix(X);
        double[] target = ModelUtils.targetVector(y, matrix.length);
        double[] distinct = ModelUtils.classesOf(y);
        if (distinct.length < 2) {
            throw new IllegalArgumentException("SVM needs at least two classes");
        }

        double[][] kernelMatrix = computeKernelMatrix(matrix);
        int models = distinct.length == 2 ? 1 : distinct.length;
        int[][] newSupportIndices = new int[models][];
        double[][] newSupportCoefficients = new double[models][];
        double[] newBiases = new double[models];

        for (int k = 0; k < models; k++) {
            double[] signs = new double[target.length];
            if (models == 1) {
                for (int i = 0; i < target.length; i++) {
                    signs[i] = target[i] == distinct[1] ? 1.0 : -1.0;
                }
            } else {
                for (int i = 0; i < target.length; i++) {
                    signs[i] = target[i] == distinct[k] ? 1.0 : -1.0;
                }
            }
            SvmFit model = trainBinary(kernelMatrix, signs);
            newSupportIndices[k] = model.supportIndices;
            newSupportCoefficients[k] = model.coefficients;
            newBiases[k] = model.bias;
        }

        classes = distinct;
        trainingData = matrix;
        supportIndices = newSupportIndices;
        supportCoefficients = newSupportCoefficients;
        biases = newBiases;
        featureNames = X.names();
        fitted = true;
        return this;
    }

    private double[][] computeKernelMatrix(double[][] matrix) {
        int rows = matrix.length;
        double[][] result = new double[rows][rows];
        for (int i = 0; i < rows; i++) {
            result[i][i] = kernel.evaluate(matrix[i], matrix[i]);
            for (int j = 0; j < i; j++) {
                double value = kernel.evaluate(matrix[i], matrix[j]);
                result[i][j] = value;
                result[j][i] = value;
            }
        }
        return result;
    }

    private SvmFit trainBinary(double[][] kernelMatrix, double[] signs) {
        int n = signs.length;
        double[] alphas = new double[n];
        double[] errors = new double[n];
        double bias = 0.0;
        int passesWithoutChange = 0;

        while (passesWithoutChange < maxPasses) {
            for (int t = 0; t < n; t++) {
                double score = bias;
                for (int j = 0; j < n; j++) {
                    if (alphas[j] > 0.0) {
                        score += alphas[j] * signs[j] * kernelMatrix[j][t];
                    }
                }
                errors[t] = score - signs[t];
            }

            int changed = 0;
            for (int i = 0; i < n; i++) {
                double yi = signs[i];
                double errorI = errors[i];
                boolean violates = (yi * errorI < -tolerance && alphas[i] < c)
                        || (yi * errorI > tolerance && alphas[i] > 0.0);
                if (!violates) {
                    continue;
                }

                double kii = kernelMatrix[i][i];
                double alphaIOld = alphas[i];
                int j = -1;
                double alphaJNew = 0.0;
                double bestDifference = -1.0;
                for (int t = 0; t < n; t++) {
                    if (t == i) {
                        continue;
                    }
                    double yt = signs[t];
                    double alphaTOld = alphas[t];
                    double lower;
                    double upper;
                    if (yi != yt) {
                        lower = Math.max(0.0, alphaTOld - alphaIOld);
                        upper = Math.min(c, c + alphaTOld - alphaIOld);
                    } else {
                        lower = Math.max(0.0, alphaIOld + alphaTOld - c);
                        upper = Math.min(c, alphaIOld + alphaTOld);
                    }
                    if (lower >= upper) {
                        continue;
                    }
                    double eta = 2.0 * kernelMatrix[i][t] - kii - kernelMatrix[t][t];
                    if (eta >= 0.0) {
                        continue;
                    }
                    double candidate = alphaTOld - yt * (errorI - errors[t]) / eta;
                    candidate = Math.min(upper, Math.max(lower, candidate));
                    if (Math.abs(candidate - alphaTOld) < 1e-5) {
                        continue;
                    }
                    double difference = Math.abs(errorI - errors[t]);
                    if (difference > bestDifference) {
                        bestDifference = difference;
                        j = t;
                        alphaJNew = candidate;
                    }
                }
                if (j < 0) {
                    continue;
                }

                double yj = signs[j];
                double errorJ = errors[j];
                double alphaJOld = alphas[j];
                double kjj = kernelMatrix[j][j];
                double kij = kernelMatrix[i][j];
                double alphaINew = alphaIOld + yi * yj * (alphaJOld - alphaJNew);
                alphaINew = Math.min(c, Math.max(0.0, alphaINew));

                double b1 = bias - errorI
                        - yi * (alphaINew - alphaIOld) * kii
                        - yj * (alphaJNew - alphaJOld) * kij;
                double b2 = bias - errorJ
                        - yi * (alphaINew - alphaIOld) * kij
                        - yj * (alphaJNew - alphaJOld) * kjj;
                boolean iFree = alphaINew > 0.0 && alphaINew < c;
                boolean jFree = alphaJNew > 0.0 && alphaJNew < c;
                double newBias;
                if (iFree && jFree) {
                    newBias = (b1 + b2) / 2.0;
                } else if (iFree) {
                    newBias = b1;
                } else if (jFree) {
                    newBias = b2;
                } else {
                    newBias = (b1 + b2) / 2.0;
                }

                double deltaI = alphaINew - alphaIOld;
                double deltaJ = alphaJNew - alphaJOld;
                double deltaBias = newBias - bias;
                alphas[i] = alphaINew;
                alphas[j] = alphaJNew;
                bias = newBias;
                for (int t = 0; t < n; t++) {
                    errors[t] += yi * deltaI * kernelMatrix[i][t]
                            + yj * deltaJ * kernelMatrix[j][t] + deltaBias;
                }
                changed++;
            }
            if (changed == 0) {
                passesWithoutChange++;
            } else {
                passesWithoutChange = 0;
            }
        }

        List<Integer> indices = new ArrayList<>();
        List<Double> coefficients = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (alphas[i] > 1e-6) {
                indices.add(i);
                coefficients.add(alphas[i] * signs[i]);
            }
        }
        int[] support = new int[indices.size()];
        double[] supportCoefficients = new double[indices.size()];
        for (int i = 0; i < support.length; i++) {
            support[i] = indices.get(i);
            supportCoefficients[i] = coefficients.get(i);
        }
        return new SvmFit(support, supportCoefficients, bias);
    }

    @Override
    public double[] predict(DataFrame X) {
        ModelUtils.requireFitted(fitted, "SVM");
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
        ModelUtils.requireFitted(fitted, "SVM");
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
        double[][] decisions = new double[matrix.length][supportIndices.length];
        for (int r = 0; r < matrix.length; r++) {
            for (int k = 0; k < supportIndices.length; k++) {
                double score = biases[k];
                int[] indices = supportIndices[k];
                double[] coefficients = supportCoefficients[k];
                for (int s = 0; s < indices.length; s++) {
                    score += coefficients[s] * kernel.evaluate(trainingData[indices[s]], matrix[r]);
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
        ModelUtils.requireFitted(fitted, "SVM");
        return classes.clone();
    }

    /** Number of support vectors associated with each one-vs-rest model. */
    public int[] supportVectorCounts() {
        ModelUtils.requireFitted(fitted, "SVM");
        int[] counts = new int[supportIndices.length];
        for (int k = 0; k < supportIndices.length; k++) {
            counts[k] = supportIndices[k].length;
        }
        return counts;
    }

    public double[] biases() {
        ModelUtils.requireFitted(fitted, "SVM");
        return biases.clone();
    }

    public Kernel kernel() {
        return kernel;
    }

    public double c() {
        return c;
    }

    public double tolerance() {
        return tolerance;
    }

    public int maxPasses() {
        return maxPasses;
    }

    private record SvmFit(int[] supportIndices, double[] coefficients, double bias) {
    }
}

