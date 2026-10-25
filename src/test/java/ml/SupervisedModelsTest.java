package ml;

import ml.kernel.Kernel;
import ml.model.DecisionTree;
import ml.model.LinearRegression;
import ml.model.LogisticRegression;
import ml.model.RandomForest;
import ml.model.SVM;
import ml.preprocessing.StandardScaler;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class SupervisedModelsTest {

    @Test
    void linearRegressionRecoversLinearRelationship() {
        int n = 21;
        double[][] features = new double[n][1];
        double[] targets = new double[n];
        for (int i = 0; i < n; i++) {
            features[i][0] = -1.0 + i * 0.1;
            targets[i] = 3.0 * features[i][0] + 2.0;
        }
        DataFrame X = DataFrame.ofNumeric(new String[]{"x"}, features);
        Vector y = Vector.of(targets);

        LinearRegression model = new LinearRegression(0.1, 5_000, 0.0, 1e-12, true);
        model.fit(X, y);

        assertTrue(model.score(X, y) > 0.999);
        assertEquals(2.0, model.intercept(), 0.02);
        assertEquals(3.0, model.coefficients()[0], 0.02);
        double[][] probabilities = model.predictProbabilities(X);
        assertEquals(1, probabilities[0].length);
        assertEquals(model.predict(X)[0], probabilities[0][0], 1e-12);
    }

    @Test
    void linearRegressionRequiresFitting() {
        LinearRegression model = new LinearRegression();
        DataFrame X = DataFrame.ofNumeric(new String[]{"x"}, new double[][]{{1.0}});
        assertThrows(IllegalStateException.class, () -> model.predict(X));
    }

    @Test
    void logisticRegressionSeparatesBinaryClasses() {
        int n = 80;
        double[][] features = new double[n][1];
        double[] targets = new double[n];
        for (int i = 0; i < n; i++) {
            features[i][0] = -2.0 + 4.0 * i / (n - 1);
            targets[i] = features[i][0] >= 0.0 ? 1.0 : 0.0;
        }
        DataFrame X = DataFrame.ofNumeric(new String[]{"x"}, features);
        Vector y = Vector.of(targets);

        LogisticRegression model = new LogisticRegression(0.5, 3_000, 0.0, 1e-10, true);
        model.fit(X, y);

        assertTrue(model.score(X, y) > 0.95);
        assertArrayEquals(new double[]{0.0, 1.0}, model.classes());
        double[][] probabilities = model.predictProbabilities(X);
        assertEquals(2, probabilities[0].length);
        for (double[] row : probabilities) {
            assertEquals(1.0, row[0] + row[1], 1e-9);
        }
        assertTrue(model.predictProbabilities(DataFrame.ofNumeric(new String[]{"x"},
                new double[][]{{2.0}}))[0][1] > 0.5);
    }

    @Test
    void logisticRegressionHandlesMultipleClasses() {
        int n = 90;
        double[][] features = new double[n][1];
        double[] targets = new double[n];
        for (int i = 0; i < n; i++) {
            double x = -3.0 + 6.0 * i / (n - 1);
            features[i][0] = x;
            targets[i] = x < -1.0 ? 0.0 : (x <= 1.0 ? 1.0 : 2.0);
        }
        DataFrame X = DataFrame.ofNumeric(new String[]{"x"}, features);
        Vector y = Vector.of(targets);

        LogisticRegression model = new LogisticRegression(0.5, 3_000, 0.0, 1e-10, true);
        model.fit(X, y);

        assertTrue(model.score(X, y) > 0.90);
        assertEquals(3, model.classes().length);
        double[][] probabilities = model.predictProbabilities(X);
        assertEquals(3, probabilities[0].length);
        for (double[] row : probabilities) {
            assertEquals(1.0, row[0] + row[1] + row[2], 1e-9);
        }
    }

    @Test
    void linearSvmSeparatesTwoClusters() {
        int n = 40;
        double[][] features = new double[n][2];
        double[] targets = new double[n];
        for (int i = 0; i < n; i++) {
            double sign = i < n / 2 ? -1.0 : 1.0;
            features[i][0] = sign * (1.0 + 0.1 * (i % 5));
            features[i][1] = 0.2 * (i % 3);
            targets[i] = i < n / 2 ? 0.0 : 1.0;
        }
        DataFrame X = DataFrame.ofNumeric(new String[]{"x1", "x2"}, features);
        Vector y = Vector.of(targets);

        SVM model = new SVM(10.0, Kernel.linear(), 1e-3, 20);
        model.fit(X, y);
        assertTrue(model.score(X, y) >= 0.95);
        assertTrue(model.supportVectorCounts()[0] > 0);
        double[][] probabilities = model.predictProbabilities(X);
        assertEquals(1.0, probabilities[0][0] + probabilities[0][1], 1e-9);
    }

    @Test
    void linearSvmConvergesOnLargerLinearProblem() {
        Random random = new Random(123);
        int n = 200;
        int d = 4;
        double[][] features = new double[n][d];
        double[] targets = new double[n];
        for (int i = 0; i < n; i++) {
            double latent = 0.0;
            for (int j = 0; j < d; j++) {
                features[i][j] = random.nextGaussian();
                latent += (j - 1.5) * features[i][j];
            }
            targets[i] = latent + 0.2 > 0 ? 1.0 : 0.0;
        }
        DataFrame X = DataFrame.ofNumeric(new String[]{"a", "b", "c", "d"}, features);
        Vector y = Vector.of(targets);

        SVM model = new SVM(10.0, Kernel.linear(), 1e-4, 100);
        model.fit(X, y);
        assertTrue(model.score(X, y) >= 0.98);
    }

    @Test
    void rbfSvmLearnsXorPattern() {
        double[][] features = {
                {-1, -1}, {-1, 1}, {1, -1}, {1, 1},
                {-2, -1}, {-1, -2}, {1, 2}, {2, 1}
        };
        double[] targets = {0, 1, 1, 0, 0, 0, 0, 0};
        DataFrame X = DataFrame.ofNumeric(new String[]{"x1", "x2"}, features);
        Vector y = Vector.of(targets);

        SVM model = new SVM(10.0, Kernel.rbf(1.0), 1e-3, 20);
        model.fit(X, y);
        assertTrue(model.score(X, y) >= 0.9);
    }

    @Test
    void decisionTreeLearnsInteractionAndReportsProbabilities() {
        DataFrame X = DataFrame.ofNumeric(new String[]{"x1", "x2"}, new double[][]{
                {0, 0}, {0, 1}, {1, 0}, {1, 1}
        });
        Vector y = Vector.of(0, 0, 0, 1);

        for (DecisionTree.Criterion criterion : DecisionTree.Criterion.values()) {
            DecisionTree tree = new DecisionTree(criterion, 3, 2, 1, DecisionTree.ALL_FEATURES, 42L);
            tree.fit(X, y);
            assertArrayEquals(new double[]{0, 0, 0, 1}, tree.predict(X));
            assertEquals(1.0, tree.score(X, y));

            double[][] probabilities = tree.predictProbabilities(X);
            assertEquals(2, probabilities[0].length);
            assertEquals(1.0, probabilities[0][0] + probabilities[0][1], 1e-9);
            assertEquals(1.0, probabilities[3][1], 1e-9);

            double sum = 0.0;
            for (double importance : tree.featureImportances()) {
                sum += importance;
            }
            assertEquals(1.0, sum, 1e-9);
        }
    }

    @Test
    void randomForestClassifiesAndAveragesProbabilities() {
        int n = 120;
        double[][] features = new double[n][2];
        double[] targets = new double[n];
        for (int i = 0; i < n; i++) {
            double x = -3.0 + 6.0 * i / (n - 1);
            features[i][0] = x;
            features[i][1] = (i % 7) * 0.1;
            targets[i] = x > 0 ? 1.0 : 0.0;
        }
        DataFrame X = DataFrame.ofNumeric(new String[]{"x1", "x2"}, features);
        Vector y = Vector.of(targets);

        RandomForest forest = new RandomForest(40, DecisionTree.Criterion.GINI, 6, 2, 1,
                DecisionTree.ALL_FEATURES, 7L);
        forest.fit(X, y);

        assertTrue(forest.score(X, y) > 0.95);
        double[][] probabilities = forest.predictProbabilities(X);
        for (double[] row : probabilities) {
            assertEquals(1.0, row[0] + row[1], 1e-9);
        }
        assertEquals(40, forest.numberOfTrees());
        double sum = 0.0;
        for (double importance : forest.featureImportances()) {
            sum += importance;
        }
        assertEquals(1.0, sum, 1e-9);
    }

    @Test
    void splitIsAlignedAndRepeatable() {
        DataFrame X = DataFrame.ofNumeric(new String[]{"x"}, new double[][]{
                {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}
        });
        Vector y = Vector.of(0, 10, 20, 30, 40, 50, 60, 70, 80, 90);

        DatasetSplit.Split first = DatasetSplit.split(X, y, 0.3, 11L);
        DatasetSplit.Split second = DatasetSplit.split(X, y, 0.3, 11L);

        assertEquals(7, first.trainSize());
        assertEquals(3, first.testSize());
        for (int i = 0; i < first.trainSize(); i++) {
            assertEquals(first.train().getDouble(i, 0) * 10, first.yTrain().get(i));
        }
        for (int i = 0; i < first.testSize(); i++) {
            assertEquals(first.test().getDouble(i, 0) * 10, first.yTest().get(i));
        }
        assertArrayEquals(first.test().numericColumn(0), second.test().numericColumn(0));
    }

    @Test
    void preprocessingPipelineImprovesRegression() {
        int n = 60;
        double[][] features = new double[n][2];
        double[] targets = new double[n];
        for (int i = 0; i < n; i++) {
            double x1 = 1000.0 + i;
            double x2 = 0.001 * i;
            features[i][0] = x1;
            features[i][1] = x2;
            targets[i] = 2.0 * (x1 - 1000.0) + 50.0 * x2 + 5.0;
        }
        DataFrame X = DataFrame.ofNumeric(new String[]{"x1", "x2"}, features);
        Vector y = Vector.of(targets);

        Pipeline pipeline = Pipeline.of(new StandardScaler());
        DataFrame transformed = pipeline.fitTransform(X);
        LinearRegression model = new LinearRegression(0.1, 5_000, 0.0, 1e-12, true);
        model.fit(transformed, y);

        assertTrue(model.score(transformed, y) > 0.999);
    }
}
