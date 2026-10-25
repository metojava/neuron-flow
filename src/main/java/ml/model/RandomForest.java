package ml.model;

import ml.DataFrame;
import ml.Estimator;
import ml.Predictor;
import ml.Vector;
import ml.metrics.Metrics;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Bootstrap-aggregated ensemble of {@link DecisionTree}s.
 *
 * <p>Each tree is trained on a bootstrap sample and considers a random subset
 * of features at every split. Prediction averages the class distributions of
 * all trees, which yields smoother probabilities than a single tree. The
 * default feature subset size is {@code round(sqrt(numberOfFeatures))}.
 */
public final class RandomForest implements Estimator<RandomForest>, Predictor {

    private final int numberOfTrees;
    private final DecisionTree.Criterion criterion;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int minSamplesLeaf;
    private final int maxFeatures;
    private final long seed;

    private DecisionTree[] trees;
    private int[][] classMappings;
    private double[] classes;
    private double[] importances;
    private String[] featureNames;
    private boolean fitted;

    public RandomForest() {
        this(100, DecisionTree.Criterion.GINI, 8, 2, 1, DecisionTree.ALL_FEATURES, 42L);
    }

    public RandomForest(int numberOfTrees, long seed) {
        this(numberOfTrees, DecisionTree.Criterion.GINI, 8, 2, 1, DecisionTree.ALL_FEATURES, seed);
    }

    /**
     * @param numberOfTrees   number of bootstrap trees
     * @param criterion       split criterion
     * @param maxDepth        maximum depth of each tree
     * @param minSamplesSplit minimum samples required to split a tree node
     * @param minSamplesLeaf  minimum samples required in each child
     * @param maxFeatures     features considered per split; {@code 0} selects
     *                        {@code round(sqrt(features))}
     * @param seed            seed for bootstrap sampling and feature selection
     */
    public RandomForest(int numberOfTrees, DecisionTree.Criterion criterion, int maxDepth,
                        int minSamplesSplit, int minSamplesLeaf, int maxFeatures, long seed) {
        if (numberOfTrees <= 0) {
            throw new IllegalArgumentException("numberOfTrees must be positive");
        }
        if (criterion == null) {
            throw new IllegalArgumentException("criterion must not be null");
        }
        if (maxDepth < 0) {
            throw new IllegalArgumentException("maxDepth must be non-negative");
        }
        if (minSamplesSplit < 2) {
            throw new IllegalArgumentException("minSamplesSplit must be at least 2");
        }
        if (minSamplesLeaf < 1) {
            throw new IllegalArgumentException("minSamplesLeaf must be at least 1");
        }
        if (maxFeatures < 0) {
            throw new IllegalArgumentException("maxFeatures must be non-negative");
        }
        this.numberOfTrees = numberOfTrees;
        this.criterion = criterion;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.seed = seed;
    }

    @Override
    public RandomForest fit(DataFrame X, Vector y) {
        double[][] matrix = ModelUtils.numericMatrix(X);
        double[] target = ModelUtils.targetVector(y, matrix.length);
        double[] distinct = ModelUtils.classesOf(y);
        int features = matrix[0].length;
        int resolvedMaxFeatures = maxFeatures == DecisionTree.ALL_FEATURES
                ? Math.max(1, (int) Math.round(Math.sqrt(features)))
                : Math.min(maxFeatures, features);

        Random random = new Random(seed);
        DecisionTree[] forest = new DecisionTree[numberOfTrees];
        for (int t = 0; t < numberOfTrees; t++) {
            int[] bootstrap = new int[matrix.length];
            for (int i = 0; i < bootstrap.length; i++) {
                bootstrap[i] = random.nextInt(matrix.length);
            }
            DecisionTree tree = new DecisionTree(criterion, maxDepth, minSamplesSplit,
                    minSamplesLeaf, resolvedMaxFeatures, random.nextLong());
            tree.fit(X.takeRows(bootstrap), y.take(bootstrap));
            forest[t] = tree;
        }

        int[][] mappings = new int[forest.length][];
        for (int t = 0; t < forest.length; t++) {
            double[] treeClasses = forest[t].classes();
            Map<Double, Integer> lookup = new HashMap<>();
            for (int k = 0; k < distinct.length; k++) {
                lookup.put(distinct[k], k);
            }
            mappings[t] = new int[treeClasses.length];
            for (int k = 0; k < treeClasses.length; k++) {
                mappings[t][k] = lookup.get(treeClasses[k]);
            }
        }

        double[] averagedImportances = new double[features];
        for (DecisionTree tree : forest) {
            double[] treeImportances = tree.featureImportances();
            for (int c = 0; c < features; c++) {
                averagedImportances[c] += treeImportances[c];
            }
        }
        for (int c = 0; c < features; c++) {
            averagedImportances[c] /= forest.length;
        }

        trees = forest;
        classMappings = mappings;
        classes = distinct;
        importances = averagedImportances;
        featureNames = X.names();
        fitted = true;
        return this;
    }

    @Override
    public double[] predict(DataFrame X) {
        ModelUtils.requireFitted(fitted, "RandomForest");
        double[][] probabilities = averageProbabilities(ModelUtils.predictionMatrix(X, featureNames));
        double[] predictions = new double[probabilities.length];
        for (int r = 0; r < probabilities.length; r++) {
            predictions[r] = classes[ModelUtils.argmax(probabilities[r])];
        }
        return predictions;
    }

    @Override
    public double[][] predictProbabilities(DataFrame X) {
        ModelUtils.requireFitted(fitted, "RandomForest");
        return averageProbabilities(ModelUtils.predictionMatrix(X, featureNames));
    }

    private double[][] averageProbabilities(double[][] matrix) {
        double[][] result = new double[matrix.length][classes.length];
        for (int t = 0; t < trees.length; t++) {
            double[][] treeProbabilities = trees[t].probabilityMatrix(matrix);
            int[] mapping = classMappings[t];
            for (int r = 0; r < matrix.length; r++) {
                for (int k = 0; k < mapping.length; k++) {
                    result[r][mapping[k]] += treeProbabilities[r][k];
                }
            }
        }
        double inverseTrees = 1.0 / trees.length;
        for (double[] row : result) {
            for (int k = 0; k < row.length; k++) {
                row[k] *= inverseTrees;
            }
        }
        return result;
    }

    /** Accuracy on the supplied data. */
    public double score(DataFrame X, Vector y) {
        return Metrics.accuracy(y, predict(X));
    }

    /** Class labels in ascending order; probability columns follow this order. */
    public double[] classes() {
        ModelUtils.requireFitted(fitted, "RandomForest");
        return classes.clone();
    }

    /** Averaged normalised feature importances across the ensemble. */
    public double[] featureImportances() {
        ModelUtils.requireFitted(fitted, "RandomForest");
        return importances.clone();
    }

    /** Number of individual trees in the fitted forest. */
    public int numberOfTrees() {
        return numberOfTrees;
    }

    public DecisionTree.Criterion criterion() {
        return criterion;
    }

    public int maxDepth() {
        return maxDepth;
    }

    public int minSamplesSplit() {
        return minSamplesSplit;
    }

    public int minSamplesLeaf() {
        return minSamplesLeaf;
    }

    public int maxFeatures() {
        return maxFeatures;
    }

    public long seed() {
        return seed;
    }
}
