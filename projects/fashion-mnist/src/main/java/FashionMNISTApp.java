import activations.ActivationFunction;
import data.FashionMNISTDownloader;
import data.FashionMNISTLoader;
import data.FashionMNISTLoader.Dataset;
import evaluation.ClassificationMetrics;
import evaluation.HtmlReport;
import initialization.WeightInitializer;
import layers.NeuralNetwork;
import layers.ModelFormat;
import losses.LossFunction;
import optim.Adam;
import training.Trainer;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Standalone Fashion-MNIST demo using the neuronflow
 * library. Trains a Flatten -> Dense(128, ReLU) -> Dense(10)
 * MLP with Adam and softmax cross-entropy, logging loss,
 * accuracy and wall-clock time every epoch, then prints a
 * confusion matrix and per-class precision / recall / F1.
 * Run with: mvn exec:exec (after mvn install in the root).
 */
public class FashionMNISTApp {

    private static final int[] TOPOLOGY = {784, 128, 10};

    private static final ActivationFunction[] ACTIVATIONS = {
        ActivationFunction.linear(),
        ActivationFunction.relu(),
        ActivationFunction.linear()
    };

    private static final float LEARNING_RATE = 0.001f;
    private static final long SEED = 123L;
    private static final int TRAIN_LIMIT = 10000;
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;

    public static void main(String[] args) throws Exception {
        if (args.length > 0 && "--cnn".equals(args[0])) {
            FashionCNNApp.main(java.util.Arrays.copyOfRange(
                args, 1, args.length));
            return;
        }
        int batchSize = args.length > 0
            ? Integer.parseInt(args[0]) : BATCH_SIZE;

        if (batchSize <= 0) {
            System.err.println(
                "Batch size must be positive, got " + batchSize);
            System.exit(1);
        }

        Path dir = FashionMNISTDownloader.ensureDownloaded();
        Dataset data = FashionMNISTLoader.load(dir);

        System.out.println("Train: " + data.trainImages().length
            + "  Test: " + data.testImages().length);
        System.out.println("Batch size: " + batchSize);

        int trainSize = Math.min(TRAIN_LIMIT,
            data.trainImages().length);
        float[][] x = new float[trainSize][];
        float[][] yAll = data.trainTargetsOneHot();
        float[][] y = new float[trainSize][];

        for (int i = 0; i < trainSize; i++) {
            x[i] = data.trainImages()[i];
            y[i] = yAll[i];
        }

        System.out.println("Examples used for training: "
            + trainSize);

        NeuralNetwork net = new NeuralNetwork(
            TOPOLOGY, ACTIVATIONS,
            LossFunction.softmaxCrossEntropy(),
            new Adam(LEARNING_RATE), SEED,
            WeightInitializer.heNormal());

        System.out.println();
        System.out.println("Initial training loss: "
            + String.format("%.6f",
                net.calculateLoss(
                    slice(x, 0, 1000), slice(y, 0, 1000))));

        System.out.println();
        System.out.println("Training (Keras-like model)...");
        System.out.println();
        System.out.printf("%-10s %-13s %-12s %s%n",
            "Epoch", "Train loss", "Accuracy", "Time");
        System.out.println(
            "-------------------------------------------");

        long startTime = System.currentTimeMillis();

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            long epochStart = System.currentTimeMillis();

            net.fit(new Trainer(1, batchSize, true,
                    SEED + epoch), x, y);

            long elapsed =
                System.currentTimeMillis() - epochStart;
            float loss = net.calculateLoss(
                slice(x, 0, 1000), slice(y, 0, 1000));
            float acc = accuracy(net, x,
                data.trainLabels(), trainSize);

            System.out.printf("%-10d %-13.6f %-12s %s%n",
                epoch, loss,
                String.format("%.2f%%", acc * 100),
                String.format("%.1f s", elapsed / 1000.0));
        }

        long totalTime =
            System.currentTimeMillis() - startTime;

        System.out.println();
        System.out.println("Training completed in "
            + String.format("%.1f", totalTime / 1000.0)
            + " s");
        System.out.println("Final training loss: "
            + String.format("%.6f",
                net.calculateLoss(
                    slice(x, 0, 1000), slice(y, 0, 1000))));

        // Full evaluation on the test set.
        int[] predicted = predictAll(net, data.testImages());
        int[] labels = data.testLabels();
        int correct = 0;

        for (int i = 0; i < labels.length; i++) {
            if (predicted[i] == labels[i]) {
                correct++;
            }
        }

        float testAcc = (float) correct / labels.length;

        System.out.println();
        System.out.println("Final evaluation:");
        System.out.printf("Test accuracy: %.2f%% (%d/%d)%n",
            testAcc * 100, correct, labels.length);

        int[][] confusion = ClassificationMetrics
            .confusionMatrix(predicted, labels,
                FashionMNISTLoader.NUM_CLASSES);

        System.out.println();
        System.out.println("Confusion matrix "
            + "(rows = actual, cols = predicted):");
        ClassificationMetrics.printReport(System.out,
            confusion, FashionMNISTLoader.CLASS_NAMES);

        // Save the trained model in all three formats.
        Path outDir = Path.of("model");
        Files.createDirectories(outDir);

        net.saveModel(outDir.resolve("fashion.bin"),
            ModelFormat.BINARY);
        net.saveModel(outDir.resolve("fashion.json"),
            ModelFormat.JSON);
        net.saveModel(outDir.resolve("fashion.csv"),
            ModelFormat.CSV);

        System.out.println();
        System.out.println("Model saved to " + outDir
            + " as .bin / .json / .csv");

        // Write the HTML evaluation report with embedded images.
        Path report = Path.of("fashion-eval.html");

        HtmlReport.write(report, confusion,
            FashionMNISTLoader.CLASS_NAMES,
            data.testImages(), labels, predicted, 2);

        System.out.println("HTML report written to "
            + report.toAbsolutePath());

        // Sample predictions.
        System.out.println();
        System.out.println("Sample predictions:");

        for (int i = 0; i < 15; i++) {
            int idx = (int) (Math.random()
                * labels.length);
            int p = predicted[idx];
            int a = labels[idx];
            System.out.printf(
                "-> Predicted: %-15s (%d) | Actual: %-15s (%d)%n",
                FashionMNISTLoader.CLASS_NAMES[p], p,
                FashionMNISTLoader.CLASS_NAMES[a], a);
        }
    }

    private static float accuracy(NeuralNetwork net,
                                  float[][] images,
                                  int[] labels,
                                  int limit) {
        int correct = 0;

        for (int i = 0; i < limit; i++) {
            if (ClassificationMetrics.argmax(
                    net.predict(images[i])) == labels[i]) {
                correct++;
            }
        }

        return (float) correct / limit;
    }

    private static int[] predictAll(NeuralNetwork net,
                                    float[][] images) {
        int[] result = new int[images.length];

        for (int i = 0; i < images.length; i++) {
            result[i] = ClassificationMetrics.argmax(
                net.predict(images[i]));
        }

        return result;
    }

    private static float[][] slice(float[][] data,
                                   int from, int to) {
        float[][] result = new float[to - from][];
        System.arraycopy(data, from, result, 0, to - from);
        return result;
    }
}
