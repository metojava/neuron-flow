package ml.model;

import ml.Clusterer;
import ml.DataFrame;

import java.util.Arrays;
import java.util.Random;

/**
 * K-means clustering with k-means++ seeding and multiple restarts.
 *
 * <p>The best restart is selected by within-cluster sum of squared distances
 * (inertia). Empty clusters are repaired by moving them onto the currently
 * worst-fit point. Distances are Euclidean; standardise features first when
 * their units differ.
 */
public final class KMeans implements Clusterer {

    private final int k;
    private final int maxIterations;
    private final double tolerance;
    private final int restarts;
    private final long seed;

    private double[][] centers;
    private String[] featureNames;
    private double inertia;
    private int iterationsRun;
    private boolean fitted;

    public KMeans(int k) {
        this(k, 300, 1e-4, 10, 42L);
    }

    public KMeans(int k, int maxIterations, int restarts, long seed) {
        this(k, maxIterations, 1e-4, restarts, seed);
    }

    /**
     * @param k             number of clusters
     * @param maxIterations maximum Lloyd iterations per restart
     * @param tolerance     convergence threshold on the largest centre shift
     * @param restarts      number of k-means++ initialisations
     * @param seed          random seed for initialisation
     */
    public KMeans(int k, int maxIterations, double tolerance, int restarts, long seed) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        if (maxIterations <= 0) {
            throw new IllegalArgumentException("maxIterations must be positive");
        }
        if (!Double.isFinite(tolerance) || tolerance <= 0.0) {
            throw new IllegalArgumentException("tolerance must be positive and finite");
        }
        if (restarts <= 0) {
            throw new IllegalArgumentException("restarts must be positive");
        }
        this.k = k;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.restarts = restarts;
        this.seed = seed;
    }

    @Override
    public KMeans fit(DataFrame X) {
        double[][] matrix = ModelUtils.numericMatrix(X);
        if (matrix.length < k) {
            throw new IllegalArgumentException("k=" + k + " exceeds the number of rows (" + matrix.length + ')');
        }

        Random random = new Random(seed);
        double[][] bestCenters = null;
        double bestInertia = Double.POSITIVE_INFINITY;
        int bestIterations = 0;
        for (int restart = 0; restart < restarts; restart++) {
            double[][] initial = kMeansPlusPlus(matrix, random);
            Result result = iterate(matrix, initial);
            if (result.inertia < bestInertia) {
                bestInertia = result.inertia;
                bestCenters = result.centers;
                bestIterations = result.iterations;
            }
        }

        centers = bestCenters;
        inertia = bestInertia;
        iterationsRun = bestIterations;
        featureNames = X.names();
        fitted = true;
        return this;
    }

    private double[][] kMeansPlusPlus(double[][] matrix, Random random) {
        int rows = matrix.length;
        int features = matrix[0].length;
        double[][] result = new double[k][features];
        result[0] = matrix[random.nextInt(rows)].clone();

        double[] closest = new double[rows];
        Arrays.fill(closest, Double.POSITIVE_INFINITY);
        for (int c = 1; c < k; c++) {
            double total = 0.0;
            for (int r = 0; r < rows; r++) {
                closest[r] = Math.min(closest[r], ModelUtils.squaredDistance(matrix[r], result[c - 1]));
                total += closest[r];
            }
            int chosen = random.nextInt(rows);
            if (total > 0.0) {
                double target = random.nextDouble() * total;
                double cumulative = 0.0;
                for (int r = 0; r < rows; r++) {
                    cumulative += closest[r];
                    if (cumulative >= target) {
                        chosen = r;
                        break;
                    }
                }
            }
            result[c] = matrix[chosen].clone();
        }
        return result;
    }

    private Result iterate(double[][] matrix, double[][] initial) {
        int rows = matrix.length;
        int features = matrix[0].length;
        double[][] centers = new double[k][];
        for (int c = 0; c < k; c++) {
            centers[c] = initial[c].clone();
        }
        int[] labels = new int[rows];
        double[] closestDistances = new double[rows];
        int completed = 0;

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            assign(matrix, centers, labels, closestDistances);
            double[][] updated = new double[k][features];
            int[] counts = new int[k];
            for (int r = 0; r < rows; r++) {
                int label = labels[r];
                counts[label]++;
                for (int f = 0; f < features; f++) {
                    updated[label][f] += matrix[r][f];
                }
            }
            for (int c = 0; c < k; c++) {
                if (counts[c] > 0) {
                    for (int f = 0; f < features; f++) {
                        updated[c][f] /= counts[c];
                    }
                }
            }

            boolean[] used = new boolean[rows];
            for (int c = 0; c < k; c++) {
                if (counts[c] == 0) {
                    int farthest = -1;
                    double farthestDistance = -1.0;
                    for (int r = 0; r < rows; r++) {
                        if (!used[r] && closestDistances[r] > farthestDistance) {
                            farthestDistance = closestDistances[r];
                            farthest = r;
                        }
                    }
                    if (farthest >= 0) {
                        used[farthest] = true;
                        updated[c] = matrix[farthest].clone();
                    }
                }
            }

            double maxShift = 0.0;
            for (int c = 0; c < k; c++) {
                maxShift = Math.max(maxShift, ModelUtils.squaredDistance(centers[c], updated[c]));
            }
            centers = updated;
            completed = iteration + 1;
            if (Math.sqrt(maxShift) <= tolerance) {
                break;
            }
        }

        double finalInertia = assign(matrix, centers, labels, closestDistances);
        return new Result(centers, finalInertia, completed);
    }

    private static double assign(double[][] matrix, double[][] centers, int[] labels, double[] closestDistances) {
        double inertia = 0.0;
        for (int r = 0; r < matrix.length; r++) {
            int best = 0;
            double bestDistance = ModelUtils.squaredDistance(matrix[r], centers[0]);
            for (int c = 1; c < centers.length; c++) {
                double distance = ModelUtils.squaredDistance(matrix[r], centers[c]);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    best = c;
                }
            }
            labels[r] = best;
            closestDistances[r] = bestDistance;
            inertia += bestDistance;
        }
        return inertia;
    }

    @Override
    public int[] predict(DataFrame X) {
        ModelUtils.requireFitted(fitted, "KMeans");
        double[][] matrix = ModelUtils.predictionMatrix(X, featureNames);
        int[] labels = new int[matrix.length];
        assign(matrix, centers, labels, new double[matrix.length]);
        return labels;
    }

    /** Within-cluster sum of squared distances for the selected restart. */
    public double inertia() {
        ModelUtils.requireFitted(fitted, "KMeans");
        return inertia;
    }

    /** Cluster centres with shape {@code [k][features]}. */
    public double[][] clusterCenters() {
        ModelUtils.requireFitted(fitted, "KMeans");
        double[][] copy = new double[centers.length][];
        for (int c = 0; c < centers.length; c++) {
            copy[c] = centers[c].clone();
        }
        return copy;
    }

    /** Number of Lloyd iterations executed by the best restart. */
    public int iterationsRun() {
        ModelUtils.requireFitted(fitted, "KMeans");
        return iterationsRun;
    }

    public int numberOfClusters() {
        return k;
    }

    public int maxIterations() {
        return maxIterations;
    }

    public double tolerance() {
        return tolerance;
    }

    public int restarts() {
        return restarts;
    }

    public long seed() {
        return seed;
    }

    private record Result(double[][] centers, double inertia, int iterations) {
    }
}
