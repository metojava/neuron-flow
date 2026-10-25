package examples;

import ml.DataFrame;
import ml.DatasetSplit;
import ml.Pipeline;
import ml.Predictor;
import ml.Vector;
import ml.kernel.Kernel;
import ml.metrics.Metrics;
import ml.model.DecisionTree;
import ml.model.KMeans;
import ml.model.LinearRegression;
import ml.model.LogisticRegression;
import ml.model.RandomForest;
import ml.model.SVM;
import ml.preprocessing.OneHotEncoder;
import ml.preprocessing.SimpleImputer;
import ml.preprocessing.StandardScaler;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * End-to-end demonstration of the classical machine-learning API:
 * raw mixed-type data, preprocessing pipeline, train/test split, model
 * training, evaluation and unsupervised clustering.
 *
 * <p>Run with:
 * {@code java --add-modules jdk.incubator.vector -cp target/classes examples.ClassicalMlDemo}
 */
public final class ClassicalMlDemo {

    private ClassicalMlDemo() {
    }

    public static void main(String[] args) {
        System.out.println("=== NeuronFlow classical ML demo ===");
        classificationDemo();
        regressionDemo();
        clusteringDemo();
    }

    private static void classificationDemo() {
        int rows = 400;
        Random random = new Random(42);
        double[] age = new double[rows];
        double[] income = new double[rows];
        double[] score = new double[rows];
        String[] city = new String[rows];
        String[] plan = new String[rows];
        double[] purchased = new double[rows];
        String[] cities = {"Paris", "Berlin", "London"};

        for (int i = 0; i < rows; i++) {
            age[i] = 18 + random.nextInt(53);
            income[i] = 20 + 130 * random.nextDouble();
            score[i] = 100 * random.nextDouble();
            city[i] = cities[random.nextInt(cities.length)];
            plan[i] = random.nextBoolean() ? "premium" : "basic";

            double cityEffect = switch (city[i]) {
                case "Paris" -> 0.40;
                case "Berlin" -> -0.25;
                default -> 0.0;
            };
            double planEffect = "premium".equals(plan[i]) ? 0.45 : -0.10;
            double latent = 0.04 * (income[i] - 70.0)
                    + 0.06 * (age[i] - 40.0)
                    + 0.01 * (score[i] - 50.0)
                    + cityEffect + planEffect;
            purchased[i] = latent > 0.25 ? 1.0 : 0.0;

            if (i % 29 == 0) {
                income[i] = Double.NaN;
            }
            if (i % 37 == 0) {
                age[i] = Double.NaN;
            }
            if (i % 31 == 0) {
                score[i] = Double.NaN;
            }
            if (i % 23 == 0) {
                city[i] = null;
            }
            if (i % 41 == 0) {
                plan[i] = null;
            }
        }

        DataFrame raw = DataFrame.builder(rows)
                .addNumeric("age", age)
                .addNumeric("income", income)
                .addNumeric("score", score)
                .addCategorical("city", city)
                .addCategorical("plan", plan)
                .build();
        Vector labels = Vector.of(purchased);
        DatasetSplit.Split split = DatasetSplit.split(raw, labels, 0.25, 42L);

        Pipeline pipeline = Pipeline.of(
                new SimpleImputer(SimpleImputer.Strategy.MEAN),
                new OneHotEncoder(),
                new StandardScaler());
        DataFrame train = pipeline.fitTransform(split.train());
        DataFrame test = pipeline.transform(split.test());

        System.out.println();
        System.out.println("-- Classification: predicting purchases --");
        System.out.printf(Locale.ROOT, "Raw columns: %d, engineered columns: %d, train/test: %d/%d%n",
                raw.columns(), train.columns(), split.trainSize(), split.testSize());

        LogisticRegression logistic = new LogisticRegression(0.2, 3_000, 0.0, 1e-10, true)
                .fit(train, split.yTrain());
        SVM svm = new SVM(10.0, Kernel.linear(), 1e-4, 100).fit(train, split.yTrain());
        DecisionTree tree = new DecisionTree(DecisionTree.Criterion.GINI, 6)
                .fit(train, split.yTrain());
        RandomForest forest = new RandomForest(80, DecisionTree.Criterion.GINI, 8, 2, 1,
                DecisionTree.ALL_FEATURES, 42L).fit(train, split.yTrain());

        List<NamedModel> models = List.of(
                new NamedModel("LogisticRegression", logistic),
                new NamedModel("SVM (linear kernel)", svm),
                new NamedModel("DecisionTree", tree),
                new NamedModel("RandomForest", forest));
        for (NamedModel model : models) {
            double accuracy = Metrics.accuracy(split.yTest(), model.predictor.predict(test));
            System.out.printf(Locale.ROOT, "%-24s accuracy = %.3f%n", model.name, accuracy);
        }

        double[][] probabilities = logistic.predictProbabilities(test);
        double[] predicted = logistic.predict(test);
        double[] classes = logistic.classes();
        System.out.println("Logistic probabilities for the first three test rows:");
        for (int r = 0; r < 3; r++) {
            System.out.printf(Locale.ROOT, "  actual=%.0f predicted=%.0f P(class %.0f)=%.3f%n",
                    split.yTest().get(r), predicted[r],
                    classes[probabilities[r].length - 1],
                    probabilities[r][probabilities[r].length - 1]);
        }
        System.out.println("Top random-forest features: "
                + topFeatures(train.names(), forest.featureImportances(), 4));
    }

    private static void regressionDemo() {
        int rows = 180;
        Random random = new Random(7);
        double[][] features = new double[rows][2];
        double[] prices = new double[rows];
        for (int i = 0; i < rows; i++) {
            double area = 40 + 210 * random.nextDouble();
            double bedrooms = 1 + random.nextInt(4);
            features[i][0] = area;
            features[i][1] = bedrooms;
            prices[i] = 90.0 * area + 25_000.0 * bedrooms + 15_000.0
                    + random.nextGaussian() * 8_000.0;
        }
        DataFrame raw = DataFrame.ofNumeric(new String[]{"area", "bedrooms"}, features);
        Vector target = Vector.of(prices);
        DatasetSplit.Split split = DatasetSplit.split(raw, target, 0.25, 42L);

        Pipeline pipeline = Pipeline.of(new StandardScaler());
        DataFrame train = pipeline.fitTransform(split.train());
        DataFrame test = pipeline.transform(split.test());

        LinearRegression model = new LinearRegression(0.1, 5_000, 0.0, 1e-12, true)
                .fit(train, split.yTrain());
        double[] predictions = model.predict(test);
        double r2 = Metrics.r2Score(split.yTest(), predictions);
        double rmse = Math.sqrt(Metrics.meanSquaredError(split.yTest(), predictions));

        System.out.println();
        System.out.println("-- Regression: predicting house prices --");
        System.out.printf(Locale.ROOT, "Train/test: %d/%d, R2 = %.3f, RMSE = %.0f%n",
                split.trainSize(), split.testSize(), r2, rmse);
        for (int r = 0; r < 3; r++) {
            System.out.printf(Locale.ROOT, "  actual=%.0f predicted=%.0f%n",
                    split.yTest().get(r), predictions[r]);
        }
    }

    private static void clusteringDemo() {
        int rows = 180;
        Random random = new Random(21);
        double[][] centers = {{2, 2}, {8, 3}, {4, 9}};
        double[][] points = new double[rows][2];
        for (int i = 0; i < rows; i++) {
            double[] center = centers[i / 60];
            points[i][0] = center[0] + random.nextGaussian() * 0.6;
            points[i][1] = center[1] + random.nextGaussian() * 0.6;
        }
        DataFrame raw = DataFrame.ofNumeric(new String[]{"x", "y"}, points);
        Pipeline pipeline = Pipeline.of(new StandardScaler());
        DataFrame scaled = pipeline.fitTransform(raw);

        KMeans kmeans = new KMeans(3, 300, 1e-6, 10, 42L);
        int[] labels = kmeans.fitPredict(scaled);
        int[] sizes = new int[kmeans.numberOfClusters()];
        for (int label : labels) {
            sizes[label]++;
        }

        System.out.println();
        System.out.println("-- Clustering: k-means on three blobs --");
        System.out.printf(Locale.ROOT, "Inertia = %.3f, iterations = %d, cluster sizes = %s%n",
                kmeans.inertia(), kmeans.iterationsRun(), Arrays.toString(sizes));
        double[][] fittedCenters = kmeans.clusterCenters();
        for (int c = 0; c < fittedCenters.length; c++) {
            System.out.printf(Locale.ROOT, "  cluster %d centre (standardised) = [%.2f, %.2f]%n",
                    c, fittedCenters[c][0], fittedCenters[c][1]);
        }

        DataFrame unseen = DataFrame.ofNumeric(new String[]{"x", "y"}, new double[][]{{4.1, 9.2}});
        int cluster = kmeans.predict(pipeline.transform(unseen))[0];
        System.out.println("Unseen point (4.1, 9.2) assigned to cluster " + cluster);
    }

    private static String topFeatures(String[] names, double[] importances, int limit) {
        Integer[] order = new Integer[names.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(importances[b], importances[a]));
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < Math.min(limit, order.length); i++) {
            if (i > 0) {
                result.append(", ");
            }
            result.append(names[order[i]])
                    .append(String.format(Locale.ROOT, " (%.2f)", importances[order[i]]));
        }
        return result.toString();
    }

    private record NamedModel(String name, Predictor predictor) {
    }
}



