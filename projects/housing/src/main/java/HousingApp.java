import activations.ActivationFunction;
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
 * Project 5 - Housing price regression.
 *
 * Predicts a continuous target from 13 features. Reads
 * data/housing/housing.csv (last column = target) when present,
 * otherwise builds a synthetic regression problem so the demo
 * always runs. Both features and targets are z-scored.
 */
public class HousingApp {

    private static final int FEATURES = 13;

    public static void main(String[] args) throws IOException {
        Data data = loadData();
        System.out.println("Samples: " + data.x.length
            + ", features: " + FEATURES);

        int[] order = shuffled(data.x.length, 42L);
        int cut = (int) (data.x.length * 0.8);
        int testCount = data.x.length - cut;

        float[][] trainX = new float[cut][FEATURES];
        float[][] trainY = new float[cut][1];
        float[][] testX = new float[testCount][FEATURES];
        float[][] testY = new float[testCount][1];

        for (int i = 0; i < cut; i++) {
            trainX[i] = data.x[order[i]].clone();
            trainY[i][0] = data.y[order[i]];
        }
        for (int i = 0; i < testCount; i++) {
            testX[i] = data.x[order[cut + i]].clone();
            testY[i][0] = data.y[order[cut + i]];
        }

        float[] xMean = new float[FEATURES];
        float[] xStd = new float[FEATURES];
        for (int f = 0; f < FEATURES; f++) {
            xMean[f] = mean(trainX, f);
            xStd[f] = std(trainX, f, xMean[f]);
            scale(trainX, f, xMean[f], xStd[f]);
            scale(testX, f, xMean[f], xStd[f]);
        }

        float yMean = mean1(trainY);
        float yStd = std1(trainY, yMean);
        scale1(trainY, yMean, yStd);
        scale1(testY, yMean, yStd);

        NeuralNetwork net = new NeuralNetwork(
            new int[] {FEATURES, 64, 32, 1},
            new ActivationFunction[] {
                ActivationFunction.linear(),
                ActivationFunction.leakyRelu(0.01f),
                ActivationFunction.relu(),
                ActivationFunction.linear()
            },
            LossFunction.meanSquaredError(),
            new Adam(0.001f),
            42L,
            WeightInitializer.heNormal());

        net.fit(new Trainer(200, 32, true, 42L), trainX, trainY);

        System.out.println();
        System.out.println("  actual   predicted    error");

        double sumSq = 0.0;
        for (int i = 0; i < testCount; i++) {
            float pred = net.predict(testX[i])[0] * yStd + yMean;
            float real = testY[i][0] * yStd + yMean;
            double err = pred - real;
            sumSq += err * err;
            if (i % 20 == 0) {
                System.out.printf("%9.2f %11.2f %8.2f%n",
                    real, pred, pred - real);
            }
        }
        double rmse = Math.sqrt(sumSq / testCount);
        System.out.printf("%nRMSE (standardized units): %.4f%n",
            rmse);
    }

    private static Data loadData() throws IOException {
        Path file = Paths.get("data", "housing", "housing.csv");
        if (Files.exists(file)) {
            System.out.println("Loading " + file + " ...");
            return parse(file);
        }
        System.out.println(
            "No data/housing/housing.csv found - using a synthetic "
            + "regression dataset (13 features).");
        return synthetic();
    }

    private static Data parse(Path file) throws IOException {
        List<float[]> rows = new ArrayList<>();
        for (String line : Files.readAllLines(file)) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            String[] parts = line.split(",");
            if (parts.length < FEATURES + 1) {
                continue;
            }
            float[] row = new float[FEATURES + 1];
            try {
                for (int f = 0; f <= FEATURES; f++) {
                    row[f] = Float.parseFloat(parts[f].trim());
                }
            } catch (NumberFormatException e) {
                continue;
            }
            rows.add(row);
        }
        float[][] x = new float[rows.size()][FEATURES];
        float[] y = new float[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, x[i], 0, FEATURES);
            y[i] = rows.get(i)[FEATURES];
        }
        return new Data(x, y);
    }

    private static Data synthetic() {
        Random rand = new Random(42L);
        int n = 500;
        float[][] x = new float[n][FEATURES];
        float[] y = new float[n];

        float[] weights = new float[FEATURES];
        for (int f = 0; f < FEATURES; f++) {
            weights[f] = (float) (rand.nextGaussian()
                * (f % 2 == 0 ? 3.0 : 1.0));
        }

        for (int i = 0; i < n; i++) {
            float sum = 5.0f;
            for (int f = 0; f < FEATURES; f++) {
                float range = (f % 3 == 0) ? 100.0f : 1.0f;
                x[i][f] = (float) (rand.nextGaussian() * range);
                sum += weights[f] * x[i][f];
            }
            y[i] = sum + (float) rand.nextGaussian() * 2.0f;
        }
        return new Data(x, y);
    }

    // ── stats helpers ───────────────────────────────────────

    private static float mean(float[][] data, int col) {
        float s = 0.0f;
        for (float[] row : data) {
            s += row[col];
        }
        return s / data.length;
    }

    private static float std(float[][] data, int col, float mean) {
        float s = 0.0f;
        for (float[] row : data) {
            float d = row[col] - mean;
            s += d * d;
        }
        return (float) Math.sqrt(s / data.length) + 1e-8f;
    }

    private static void scale(float[][] data, int col,
                              float mean, float std) {
        for (float[] row : data) {
            row[col] = (row[col] - mean) / std;
        }
    }

    private static float mean1(float[][] data) {
        float s = 0.0f;
        for (float[] row : data) {
            s += row[0];
        }
        return s / data.length;
    }

    private static float std1(float[][] data, float mean) {
        float s = 0.0f;
        for (float[] row : data) {
            float d = row[0] - mean;
            s += d * d;
        }
        return (float) Math.sqrt(s / data.length) + 1e-8f;
    }

    private static void scale1(float[][] data, float mean,
                               float std) {
        for (float[] row : data) {
            row[0] = (row[0] - mean) / std;
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

    private static final class Data {
        final float[][] x;
        final float[] y;

        Data(float[][] x, float[] y) {
            this.x = x;
            this.y = y;
        }
    }
}
