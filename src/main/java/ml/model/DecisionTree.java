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
 * CART-style classification tree supporting Gini impurity and entropy
 * (information gain) splits.
 *
 * <p>The tree is built depth-first and chooses, at every internal node, the
 * feature/threshold pair with the largest weighted impurity decrease. Leaves
 * store the empirical class distribution, which is reused by
 * {@link #predictProbabilities(DataFrame)}. Random feature subsets can be
 * enabled with {@code maxFeatures}, which is how {@link RandomForest} injects
 * diversity between trees.
 */
public final class DecisionTree implements Estimator<DecisionTree>, Predictor {

    /** Split quality measure. */
    public enum Criterion {
        GINI,
        ENTROPY
    }

    /** Sentinel meaning "consider every feature at every split". */
    public static final int ALL_FEATURES = 0;

    private final Criterion criterion;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int minSamplesLeaf;
    private final int maxFeatures;
    private final long seed;

    private String[] featureNames;
    private double[] classes;
    private Node root;
    private double[] importances;
    private int totalSamples;
    private boolean fitted;

    public DecisionTree() {
        this(Criterion.GINI, 10, 2, 1, ALL_FEATURES, 42L);
    }

    public DecisionTree(Criterion criterion, int maxDepth) {
        this(criterion, maxDepth, 2, 1, ALL_FEATURES, 42L);
    }

    /**
     * @param criterion        split function
     * @param maxDepth         maximum tree depth ({@code 0} produces a single leaf)
     * @param minSamplesSplit  minimum samples required to attempt a split
     * @param minSamplesLeaf   minimum samples required in each child
     * @param maxFeatures      features sampled at each split; {@code 0} means all
     * @param seed             seed used when {@code maxFeatures} is active
     */
    public DecisionTree(Criterion criterion, int maxDepth, int minSamplesSplit,
                        int minSamplesLeaf, int maxFeatures, long seed) {
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
        this.criterion = criterion;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.seed = seed;
    }

    @Override
    public DecisionTree fit(DataFrame X, Vector y) {
        double[][] matrix = ModelUtils.numericMatrix(X);
        double[] target = ModelUtils.targetVector(y, matrix.length);
        classes = ModelUtils.classesOf(y);
        Map<Double, Integer> lookup = new HashMap<>();
        for (int i = 0; i < classes.length; i++) {
            lookup.put(classes[i], i);
        }
        int[] classIndex = new int[target.length];
        for (int i = 0; i < target.length; i++) {
            classIndex[i] = lookup.get(target[i]);
        }

        featureNames = X.names();
        totalSamples = matrix.length;
        importances = new double[matrix[0].length];
        int[] indices = new int[matrix.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        root = build(matrix, classIndex, indices, 0, new Random(seed));

        double sum = 0.0;
        for (double importance : importances) {
            sum += importance;
        }
        if (sum > 0.0) {
            for (int i = 0; i < importances.length; i++) {
                importances[i] /= sum;
            }
        }
        fitted = true;
        return this;
    }

    private Node build(double[][] matrix, int[] classIndex, int[] indices,
                       int depth, Random random) {
        int[] counts = counts(classIndex, indices, classes.length);
        if (indices.length < minSamplesSplit || depth >= maxDepth) {
            return leaf(counts, indices.length);
        }
        double parentImpurity = impurity(counts, indices.length);
        if (parentImpurity <= 1e-12) {
            return leaf(counts, indices.length);
        }

        int[] candidateFeatures = selectFeatures(matrix[0].length, random);
        Split split = bestSplit(matrix, classIndex, indices, counts, parentImpurity, candidateFeatures);
        if (split == null || split.gain <= 1e-12) {
            return leaf(counts, indices.length);
        }

        int leftSize = 0;
        for (int index : indices) {
            if (matrix[index][split.feature] <= split.threshold) {
                leftSize++;
            }
        }
        int rightSize = indices.length - leftSize;
        if (leftSize < minSamplesLeaf || rightSize < minSamplesLeaf) {
            return leaf(counts, indices.length);
        }

        int[] leftIndices = new int[leftSize];
        int[] rightIndices = new int[rightSize];
        int left = 0;
        int right = 0;
        for (int index : indices) {
            if (matrix[index][split.feature] <= split.threshold) {
                leftIndices[left++] = index;
            } else {
                rightIndices[right++] = index;
            }
        }

        importances[split.feature] += ((double) indices.length / totalSamples) * split.gain;
        Node leftNode = build(matrix, classIndex, leftIndices, depth + 1, random);
        Node rightNode = build(matrix, classIndex, rightIndices, depth + 1, random);
        return new Node(split.feature, split.threshold, leftNode, rightNode, Double.NaN, null);
    }

    private Split bestSplit(double[][] matrix, int[] classIndex, int[] indices,
                            int[] nodeCounts, double parentImpurity, int[] features) {
        int nodeSize = indices.length;
        Split best = null;
        for (int feature : features) {
            int[] order = sortedByFeature(matrix, feature, indices);
            int[] leftCounts = new int[classes.length];
            for (int position = 0; position < order.length - 1; position++) {
                leftCounts[classIndex[order[position]]]++;
                int leftSize = position + 1;
                int rightSize = nodeSize - leftSize;
                if (leftSize < minSamplesLeaf || rightSize < minSamplesLeaf) {
                    continue;
                }
                double current = matrix[order[position]][feature];
                double next = matrix[order[position + 1]][feature];
                if (current >= next) {
                    continue;
                }
                double weighted = (leftSize * impurity(leftCounts, leftSize)
                        + rightSize * impurity(remaining(nodeCounts, leftCounts), rightSize)) / nodeSize;
                double gain = parentImpurity - weighted;
                if (best == null || gain > best.gain) {
                    double threshold = current + (next - current) / 2.0;
                    best = new Split(feature, threshold, gain);
                }
            }
        }
        return best;
    }

    private static int[] remaining(int[] total, int[] left) {
        int[] right = new int[total.length];
        for (int i = 0; i < total.length; i++) {
            right[i] = total[i] - left[i];
        }
        return right;
    }

    private int[] selectFeatures(int features, Random random) {
        if (maxFeatures == ALL_FEATURES || maxFeatures >= features) {
            int[] all = new int[features];
            for (int i = 0; i < features; i++) {
                all[i] = i;
            }
            return all;
        }
        int[] order = new int[features];
        for (int i = 0; i < features; i++) {
            order[i] = i;
        }
        for (int i = features - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        int[] selected = new int[maxFeatures];
        System.arraycopy(order, 0, selected, 0, maxFeatures);
        return selected;
    }

    private static int[] sortedByFeature(double[][] matrix, int feature, int[] indices) {
        int[] values = indices.clone();
        int[] buffer = new int[values.length];
        mergeSort(values, buffer, 0, values.length, matrix, feature);
        return values;
    }

    private static void mergeSort(int[] values, int[] buffer, int from, int to,
                                  double[][] matrix, int feature) {
        if (to - from < 2) {
            return;
        }
        int middle = (from + to) >>> 1;
        mergeSort(values, buffer, from, middle, matrix, feature);
        mergeSort(values, buffer, middle, to, matrix, feature);
        int i = from;
        int j = middle;
        int k = from;
        while (i < middle && j < to) {
            double left = matrix[values[i]][feature];
            double right = matrix[values[j]][feature];
            if (left <= right) {
                buffer[k++] = values[i++];
            } else {
                buffer[k++] = values[j++];
            }
        }
        while (i < middle) {
            buffer[k++] = values[i++];
        }
        while (j < to) {
            buffer[k++] = values[j++];
        }
        System.arraycopy(buffer, from, values, from, to - from);
    }

    private static int[] counts(int[] classIndex, int[] indices, int numberOfClasses) {
        int[] counts = new int[numberOfClasses];
        for (int index : indices) {
            counts[classIndex[index]]++;
        }
        return counts;
    }

    private double impurity(int[] counts, int total) {
        if (total == 0) {
            return 0.0;
        }
        double result = 0.0;
        if (criterion == Criterion.GINI) {
            for (int count : counts) {
                double probability = (double) count / total;
                result += probability * probability;
            }
            return 1.0 - result;
        }
        for (int count : counts) {
            if (count > 0) {
                double probability = (double) count / total;
                result -= probability * (Math.log(probability) / Math.log(2.0));
            }
        }
        return result;
    }

    private Node leaf(int[] counts, int total) {
        double[] distribution = new double[counts.length];
        int best = 0;
        for (int i = 0; i < counts.length; i++) {
            distribution[i] = (double) counts[i] / total;
            if (counts[i] > counts[best]) {
                best = i;
            }
        }
        return new Node(-1, Double.NaN, null, null, classes[best], distribution);
    }

    @Override
    public double[] predict(DataFrame X) {
        ModelUtils.requireFitted(fitted, "DecisionTree");
        double[][] matrix = ModelUtils.predictionMatrix(X, featureNames);
        double[] predictions = new double[matrix.length];
        for (int r = 0; r < matrix.length; r++) {
            predictions[r] = root.value(matrix[r]);
        }
        return predictions;
    }

    @Override
    public double[][] predictProbabilities(DataFrame X) {
        ModelUtils.requireFitted(fitted, "DecisionTree");
        return probabilityMatrix(ModelUtils.predictionMatrix(X, featureNames));
    }

    /** Package-private variant used by {@link RandomForest} to avoid repeated frame decoding. */
    double[][] probabilityMatrix(double[][] matrix) {
        double[][] result = new double[matrix.length][];
        for (int r = 0; r < matrix.length; r++) {
            result[r] = root.distribution(matrix[r]).clone();
        }
        return result;
    }

    /** Accuracy on the supplied data. */
    public double score(DataFrame X, Vector y) {
        return Metrics.accuracy(y, predict(X));
    }

    /** Class labels in ascending order; probability columns follow this order. */
    public double[] classes() {
        ModelUtils.requireFitted(fitted, "DecisionTree");
        return classes.clone();
    }

    /** Normalised impurity-based feature importances (may be all zeros for a stump). */
    public double[] featureImportances() {
        ModelUtils.requireFitted(fitted, "DecisionTree");
        return importances.clone();
    }

    public Criterion criterion() {
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

    private static final class Node {
        private final int feature;
        private final double threshold;
        private final Node left;
        private final Node right;
        private final double value;
        private final double[] distribution;

        private Node(int feature, double threshold, Node left, Node right,
                     double value, double[] distribution) {
            this.feature = feature;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
            this.value = value;
            this.distribution = distribution;
        }

        private boolean isLeaf() {
            return feature < 0;
        }

        private double value(double[] row) {
            Node node = this;
            while (!node.isLeaf()) {
                node = row[node.feature] <= node.threshold ? node.left : node.right;
            }
            return node.value;
        }

        private double[] distribution(double[] row) {
            Node node = this;
            while (!node.isLeaf()) {
                node = row[node.feature] <= node.threshold ? node.left : node.right;
            }
            return node.distribution;
        }
    }

    private record Split(int feature, double threshold, double gain) {
    }
}
