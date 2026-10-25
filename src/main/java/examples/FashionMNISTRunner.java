package examples;

import activations.ActivationFunction;
import data.FashionMNISTDownloader;
import data.FashionMNISTLoader;
import data.FashionMNISTLoader.Dataset;
import evaluation.ClassificationMetrics;
import initialization.WeightInitializer;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.Adam;
import training.Trainer;

import java.nio.file.Path;

/**
 * End-to-end Fashion-MNIST classification demo.
 * Downloads the dataset (if missing), trains a Keras-like
 * Flatten -> Dense(128, ReLU) -> Dense(10) MLP with Adam and
 * softmax cross-entropy, logs per-epoch loss/accuracy/time,
 * and prints a confusion matrix plus sample predictions.
 */
public class FashionMNISTRunner {

    /** Network topology: 784 inputs, 128 hidden, 10 outputs. */
    private static final int[] TOPOLOGY = {784, 128, 10};

    /** One activation per layer: linear input, ReLU hidden, linear output. */
    private static final ActivationFunction[] ACTIVATIONS = {
        ActivationFunction.linear(),
        ActivationFunction.relu(),
        ActivationFunction.linear()
    };

    /** Adam step size. */
    private static final float LEARNING_RATE = 0.001f;

    /** RNG seed for reproducibility. */
    private static final long SEED = 123L;

    /** Training examples used (subset for a quick demo). */
    private static final int TRAIN_LIMIT = 8000;

    /** Mini-batch size. */
    private static final int BATCH_SIZE = 32;

    /** Number of epochs. */
    private static final int EPOCHS = 10;

    public static void main(String[] args) throws Exception {
        System.out.println("Loading Fashion-MNIST dataset...");

        Path dataDir = FashionMNISTDownloader.ensureDownloaded();
        Dataset data = FashionMNISTLoader.load(dataDir);

        System.out.println("Training examples: "
            + data.trainImages().length);
        System.out.println("Test examples: "
            + data.testImages().length);

        // Use a subset for a fast demo.
        int trainSize = Math.min(TRAIN_LIMIT,
            data.trainImages().length);
        float[][] trainInputs = new float[trainSize][];
        float[][] trainTargets = new float[trainSize][];

        for (int i = 0; i < trainSize; i++) {
            trainInputs[i] = data.trainImages()[i];
        }

        float[][] oneHot = data.trainTargetsOneHot();
        System.arraycopy(oneHot, 0, trainTargets, 0, trainSize);

        System.out.println("Examples used for training: "
            + trainSize);

        NeuralNetwork network = new NeuralNetwork(
            TOPOLOGY, ACTIVATIONS,
            LossFunction.softmaxCrossEntropy(),
            new Adam(LEARNING_RATE),
            SEED, WeightInitializer.heNormal());

        System.out.println();
        System.out.println("Initial training loss: "
            + String.format("%.6f",
                network.calculateLoss(
                    slice(trainInputs, 0, 1000),
                    slice(trainTargets, 0, 1000))));

        System.out.println();
        System.out.println("Training (Keras-like simple model)...");
        System.out.println();
        System.out.printf("%-10s %-13s %-12s %s%n",
            "Epoch", "Train loss", "Accuracy", "Time");
        System.out.println("-------------------------------------------");

        Trainer trainer = new Trainer(EPOCHS, BATCH_SIZE,
            true, SEED);

        long startTime = System.currentTimeMillis();

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            long epochStart = System.currentTimeMillis();

            // Train for one epoch by fitting with a Trainer of 1 epoch.
            // We re-use the network's fit but need per-epoch metrics,
            // so we call fit with a single-epoch Trainer and then evaluate.
            network.fit(new Trainer(1, BATCH_SIZE, true, SEED + epoch),
                trainInputs, trainTargets);

            long elapsed = System.currentTimeMillis() - epochStart;
            float loss = network.calculateLoss(
                slice(trainInputs, 0, 1000),
                slice(trainTargets, 0, 1000));
            float acc = accuracy(network, trainInputs,
                data.trainLabels(), trainSize);

            System.out.printf("%-10d %-13.6f %-12s %s%n",
                epoch, loss,
                String.format("%.2f%%", acc * 100),
                String.format("%.1f s", elapsed / 1000.0));
        }

        long totalTime = System.currentTimeMillis() - startTime;

        System.out.println();
        System.out.println("Training completed in "
            + String.format("%.1f", totalTime / 1000.0) + " s");

        float finalLoss = network.calculateLoss(
            slice(trainInputs, 0, 1000),
            slice(trainTargets, 0, 1000));
        System.out.println("Final training loss: "
            + String.format("%.6f", finalLoss));

        // Evaluate on the full test set.
        float[][] testTargets = data.testTargetsOneHot();
        float testAcc = accuracy(network, data.testImages(),
            data.testLabels(), data.testImages().length);
        System.out.println("Test accuracy: "
            + String.format("%.2f%%", testAcc * 100));

        System.out.println();
        System.out.println("Final evaluation:");
        int correct = Math.round(testAcc * data.testLabels().length);
        System.out.println("Test accuracy: "
            + String.format("%.2f%%", testAcc * 100)
            + " (" + correct + "/" + data.testLabels().length + ")");

        // Confusion matrix + per-class metrics.
        int[] predicted = predictClasses(network,
            data.testImages());
        int[][] confusion = ClassificationMetrics
            .confusionMatrix(predicted, data.testLabels(),
                FashionMNISTLoader.NUM_CLASSES);

        System.out.println();
        System.out.println("Confusion matrix "
            + "(rows = actual, cols = predicted):");
        ClassificationMetrics.printReport(System.out,
            confusion, FashionMNISTLoader.CLASS_NAMES);

        // Sample predictions.
        System.out.println();
        System.out.println("Sample predictions:");

        for (int i = 0; i < 15; i++) {
            int idx = (int) (Math.random()
                * data.testImages().length);
            int p = predicted[idx];
            int a = data.testLabels()[idx];
            System.out.printf(
                "-> Predicted: %-15s (%d) | Actual: %-15s (%d)%n",
                FashionMNISTLoader.CLASS_NAMES[p], p,
                FashionMNISTLoader.CLASS_NAMES[a], a);
        }
    }

    /** Accuracy over the first {@code limit} rows. */
    private static float accuracy(NeuralNetwork network,
                                  float[][] images,
                                  int[] labels,
                                  int limit) {
        int correct = 0;

        for (int i = 0; i < limit; i++) {
            float[] output = network.predict(images[i]);
            int predicted = ClassificationMetrics.argmax(output);

            if (predicted == labels[i]) {
                correct++;
            }
        }

        return (float) correct / limit;
    }

    /** Predicted class for every row. */
    private static int[] predictClasses(NeuralNetwork network,
                                        float[][] images) {
        int[] result = new int[images.length];

        for (int i = 0; i < images.length; i++) {
            result[i] = ClassificationMetrics.argmax(
                network.predict(images[i]));
        }

        return result;
    }

    /** Copies rows [from, to) of a 2-D array. */
    private static float[][] slice(float[][] data,
                                   int from, int to) {
        float[][] result = new float[to - from][];
        System.arraycopy(data, from, result, 0, to - from);
        return result;
    }
}
