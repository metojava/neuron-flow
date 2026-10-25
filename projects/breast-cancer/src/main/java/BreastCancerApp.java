import activations.ActivationFunction;
import evaluation.ClassificationMetrics;
import initialization.WeightInitializer;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.Adam;
import training.Trainer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Project 4 - Breast cancer classification.
 *
 * Binary classification of 30 cell-nucleus features. Reads
 * data/breast-cancer/wdbc.data when present, otherwise builds a
 * synthetic dataset of the same shape so the demo always runs.
 */
public class BreastCancerApp {

    private static final int FEATURES = 30;

    public static void main(String[] args) throws IOException {
        Data data = loadData();

        System.out.println("Samples: " + data.x.length
            + "  benign: " + count(data.y, 0)
            + "  malignant: " + count(data.y, 1));

        int[] order = shuffled(data.x.length, 42L);
        int cut = (int) (data.x.length * 0.8);
        int testCount = data.x.length - cut;

        float[][] trainX = new float[cut][FEATURES];
        float[][] trainY = new float[cut][1];
        int[] trainLabels = new int[cut];

        float[][] testX = new float[testCount][FEATURES];
        float[][] testY = new float[testCount][1];
        int[] testLabels = new int[testCount];

        for (int i = 0; i < cut; i++) {
            trainX[i] = data.x[order[i]].clone();
            trainLabels[i] = data.y[order[i]];
            trainY[i][0] = trainLabels[i];
        }
        for (int i = 0; i < testCount; i++) {
            testX[i] = data.x[order[cut + i]].clone();
            testLabels[i] = data.y[order[cut + i]];
            testY[i][0] = testLabels[i];
        }

        standardize(trainX, testX);

        NeuralNetwork net = new NeuralNetwork(
            new int[] {FEATURES, 16, 8, 1},
            new ActivationFunction[] {
                ActivationFunction.linear(),
                ActivationFunction.relu(),
                ActivationFunction.relu(),
                ActivationFunction.sigmoid()
            },
            LossFunction.binaryCrossEntropy(),
            new Adam(0.001f),
            42L,
            WeightInitializer.heNormal());

        net.fit(new Trainer(100, 16, true, 42L), trainX, trainY);

        int[] predicted = new int[testCount];
        for (int i = 0; i < testCount; i++) {
            predicted[i] =
                net.predict(testX[i])[0] >= 0.5f ? 1 : 0;
        }

        int[][] confusion = ClassificationMetrics.confusionMatrix(
            predicted, testLabels, 2);

        System.out.printf("%nAccuracy:  %.2f%%%n",
            ClassificationMetrics.accuracy(predicted, testLabels)
                * 100.0f);
        System.out.printf("Precision: %.3f%n",
            ClassificationMetrics.precision(confusion, 1));
        System.out.printf("Recall:    %.3f%n",
            ClassificationMetrics.recall(confusion, 1));
        System.out.printf("F1:        %.3f%n",
            ClassificationMetrics.f1(confusion, 1));
        System.out.println();
        ClassificationMetrics.printReport(System.out, confusion,
            new String[] {"benign", "malignant"});
    }

    private static Data loadData() throws IOException {
        Path file = Paths.get(
            "data", "breast-cancer", "wdbc.data");
        if (Files.exists(file)) {
            System.out.println("Loading " + file + " ...");
            return parse(file);
        }
        System.out.println(
            "No data/breast-cancer/wdbc.data found - using a "
            + "synthetic dataset with the same shape (30 features).");
        return synthetic();
    }

    private static Data parse(Path file) throws IOException {
        List<float[]> features = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        for (String line : Files.readAllLines(file)) {
            line = line.trim();
            String[] parts = line.split(",");
            if (parts.length < 2 + FEATURES) {
                continue;
            }
            String diag = parts[1].trim();
            if (!diag.equalsIgnoreCase("M")
                    && !diag.equalsIgnoreCase("B")) {
                continue;
            }
            float[] row = new float[FEATURES];
            for (int f = 0; f < FEATURES; f++) {
                row[f] = Float.parseFloat(parts[2 + f].trim());
            }
            features.add(row);
            labels.add(diag.equalsIgnoreCase("M") ? 1 : 0);
        }

        float[][] x = features.toArray(new float[0][]);
        int[] y = new int[labels.size()];
        for (int i = 0; i < y.length; i++) {
            y[i] = labels.get(i);
        }
        return new Data(x, y);
    }

    private static Data synthetic() {
        Random rand = new Random(7L);
        int benign = 357;
        int malignant = 212;
        int n = benign + malignant;
        float[][] x = new float[n][FEATURES];
        int[] y = new int[n];

        for (int i = 0; i < n; i++) {
            boolean malignantCase = i >= benign;
            y[i] = malignantCase ? 1 : 0;
            for (int f = 0; f < FEATURES; f++) {
                double base = malignantCase ? 0.50 : 0.30;
                double slope = 0.25 * f / (FEATURES - 1.0);
                double mean = base + slope;
                x[i][f] = (float) (mean
                    + rand.nextGaussian() * 0.09);
            }
        }
        return new Data(x, y);
    }

    private static void standardize(float[][] train,
                                    float[][] test) {
        for (int f = 0; f < FEATURES; f++) {
            float mean = 0.0f;
            for (float[] row : train) {
                mean += row[f];
            }
            mean /= train.length;

            float var = 0.0f;
            for (float[] row : train) {
                float d = row[f] - mean;
                var += d * d;
            }
            var /= train.length;
            float std = (float) Math.sqrt(var) + 1e-8f;

            for (float[] row : train) {
                row[f] = (row[f] - mean) / std;
            }
            for (float[] row : test) {
                row[f] = (row[f] - mean) / std;
            }
        }
    }

    private static int[] shuffled(int n, long seed) {
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Random rand = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        return order;
    }

    private static int count(int[] labels, int value) {
        int c = 0;
        for (int v : labels) {
            if (v == value) {
                c++;
            }
        }
        return c;
    }

    private record Data(float[][] x, int[] y) {
    }
}
