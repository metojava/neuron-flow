import activations.ActivationFunction;
import initialization.WeightInitializer;
import layers.NeuralNetwork;
import losses.HuberLoss;
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
 * Project 9 - Stock price prediction.
 *
 * Builds 20-day sliding windows over a price series and predicts
 * the next day's close. Reads data/stock/prices.csv (one price
 * per line) when present, otherwise generates a synthetic series.
 * The temporal split never shuffles across the cut.
 */
public class StockApp {

    private static final int WINDOW = 20;
    private static final int EPOCHS = 60;

    public static void main(String[] args) throws IOException {
        float[] prices = loadSeries();
        System.out.println("Series length: " + prices.length
            + " days, window: " + WINDOW);

        float mean = mean(prices);
        float std = std(prices, mean) + 1e-8f;
        float[] series = new float[prices.length];
        for (int i = 0; i < prices.length; i++) {
            series[i] = (prices[i] - mean) / std;
        }

        List<float[]> inputs = new ArrayList<>();
        List<float[]> targets = new ArrayList<>();

        for (int i = 0; i + WINDOW < series.length; i++) {
            float[] window = new float[WINDOW];
            System.arraycopy(series, i, window, 0, WINDOW);
            inputs.add(window);
            targets.add(new float[] {series[i + WINDOW]});
        }

        int count = inputs.size();
        int cut = (int) (count * 0.8);

        float[][] trainX = new float[cut][];
        float[][] trainY = new float[cut][];
        for (int i = 0; i < cut; i++) {
            trainX[i] = inputs.get(i);
            trainY[i] = targets.get(i);
        }

        int testCount = count - cut;
        float[][] testX = new float[testCount][];
        float[][] testY = new float[testCount][];
        for (int i = 0; i < testCount; i++) {
            testX[i] = inputs.get(cut + i);
            testY[i] = targets.get(cut + i);
        }

        NeuralNetwork net = new NeuralNetwork(
            new int[] {WINDOW, 64, 32, 1},
            new ActivationFunction[] {
                ActivationFunction.linear(),
                ActivationFunction.leakyRelu(0.01f),
                ActivationFunction.tanh(),
                ActivationFunction.linear()
            },
            new HuberLoss(0.5f),
            new Adam(0.001f),
            42L,
            WeightInitializer.heNormal());

        net.fit(new Trainer(EPOCHS, 32, false, 42L),
            trainX, trainY);

        float sumAbs = 0.0f;
        int correctDirection = 0;
        for (int i = 0; i < testCount; i++) {
            float pred = net.predict(testX[i])[0];
            float actual = testY[i][0];
            float today = testX[i][WINDOW - 1];
            sumAbs += Math.abs(pred - actual);

            boolean actualUp = actual > today;
            boolean predUp = pred > today;
            if (actualUp == predUp) {
                correctDirection++;
            }
        }

        System.out.printf("%ntest MAE (normalized): %.4f%n",
            sumAbs / testCount);
        System.out.printf("directional accuracy:  %.1f%%%n",
            100.0f * correctDirection / testCount);
    }

    private static float[] loadSeries() throws IOException {
        Path file = Paths.get("data", "stock", "prices.csv");
        if (Files.exists(file)) {
            System.out.println("Loading " + file + " ...");
            List<Float> values = new ArrayList<>();
            for (String line : Files.readAllLines(file)) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                String first = line.split(",")[0].trim();
                try {
                    values.add(Float.parseFloat(first));
                } catch (NumberFormatException e) {
                    // skip header or malformed rows
                }
            }
            float[] out = new float[values.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = values.get(i);
            }
            return out;
        }
        System.out.println(
            "No data/stock/prices.csv found - using a synthetic "
            + "random-walk series with drift.");
        return synthetic();
    }

    private static float[] synthetic() {
        int days = 800;
        Random rand = new Random(11L);
        float[] prices = new float[days];
        prices[0] = 100.0f;
        for (int i = 1; i < days; i++) {
            float dailyReturn = 0.0005f
                + 0.01f * (float) rand.nextGaussian();
            prices[i] = prices[i - 1] * (1.0f + dailyReturn);
        }
        return prices;
    }

    private static float mean(float[] values) {
        float s = 0.0f;
        for (float v : values) {
            s += v;
        }
        return s / values.length;
    }

    private static float std(float[] values, float mean) {
        float s = 0.0f;
        for (float v : values) {
            float d = v - mean;
            s += d * d;
        }
        return (float) Math.sqrt(s / values.length);
    }
}
