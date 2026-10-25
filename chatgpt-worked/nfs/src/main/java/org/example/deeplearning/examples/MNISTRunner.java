package org.example.deeplearning.examples;

import org.example.deeplearning.activation.ActivationFunction;
import org.example.deeplearning.loss.LossFunction;
import org.example.deeplearning.optim.Adam;
import org.example.deeplearning.training.Trainer;
import org.example.deeplearning.core.NeuralNetwork;
import org.example.deeplearning.data.MNISTLoader;
import org.example.deeplearning.data.MNISTDownloader;

import java.nio.file.Path;
import java.util.Arrays;

public class MNISTRunner {

    private static final int INPUTS = 784;
    private static final int CLASSES = 10;

    private static final int EPOCHS = 5;
    private static final int BATCH_SIZE = 256;

    private static final long SEED = 42L;

    public static void main(String[] args) throws Exception {
        boolean quickMode = Arrays.asList(args).contains("--quick");

        Path dataDirectory = Path.of("data", "mnist");

        // 1. Download the dataset if necessary.
        MNISTDownloader.download(dataDirectory);

        // 2. Load training and test data.
        System.out.println("\nLoading MNIST dataset...");

        MNISTLoader.Dataset training =
            MNISTLoader.loadTraining(dataDirectory);

        MNISTLoader.Dataset testing =
            MNISTLoader.loadTest(dataDirectory);

        System.out.println("Training examples: " + training.size());
        System.out.println("Test examples: " + testing.size());

        // Quick mode uses a subset for an initial smoke test.
        int trainingCount = quickMode
            ? Math.min(10_000, training.size())
            : training.size();

        double[][] trainImages = Arrays.copyOf(
            training.getImages(), trainingCount);

        int[] trainLabels = Arrays.copyOf(
            training.getLabels(), trainingCount);

        double[][] trainTargets = oneHotEncode(trainLabels);

        double[][] testImages = testing.getImages();
        int[] testLabels = testing.getLabels();

        System.out.println("Examples used for training: " + trainingCount);

        // 3. Create the neural network.
        int[] topology = {INPUTS, 128, 64, CLASSES};

        ActivationFunction[] activations = {
            ActivationFunction.linear(),
            ActivationFunction.relu(),
            ActivationFunction.relu(),
            ActivationFunction.sigmoid()
        };

        NeuralNetwork network = new NeuralNetwork(
            topology,
            activations,
            LossFunction.meanSquaredError(),
            new Adam(0.001),
            SEED
        );

        System.out.printf(
            "\nInitial training loss: %.6f%n",
            network.calculateLoss(trainImages, trainTargets)
        );

        // 4. Train one epoch at a time so progress is visible.
        System.out.println("\nStarting training...");

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            Trainer trainer = new Trainer(
                1,
                BATCH_SIZE,
                true,
                SEED + epoch
            );

            long start = System.currentTimeMillis();

            network.fit(trainer, trainImages, trainTargets);

            double loss = network.calculateLoss(
                trainImages, trainTargets);

            double accuracy = evaluateAccuracy(
                network, testImages, testLabels);

            long elapsed = System.currentTimeMillis() - start;

            System.out.printf(
                "Epoch %d/%d | loss: %.6f | test accuracy: %.2f%%"
                    + " | time: %.1f s%n",
                epoch,
                EPOCHS,
                loss,
                accuracy * 100.0,
                elapsed / 1000.0
            );
        }

        // 5. Final evaluation.
        System.out.println("\nFinal evaluation:");
        evaluate(network, testImages, testLabels);

        // 6. Display sample predictions.
        System.out.println("\nSample predictions:");

        int samples = Math.min(20, testImages.length);

        for (int i = 0; i < samples; i++) {
            double[] output = network.predict(testImages[i]);
            int predicted = argmax(output);

            System.out.printf(
                "Sample %d | predicted: %d | actual: %d%n",
                i,
                predicted,
                testLabels[i]
            );
        }
    }

    private static double[][] oneHotEncode(int[] labels) {
        double[][] targets = new double[labels.length][CLASSES];

        for (int i = 0; i < labels.length; i++) {
            int label = labels[i];

            if (label < 0 || label >= CLASSES) {
                throw new IllegalArgumentException(
                    "Invalid digit label: " + label);
            }

            targets[i][label] = 1.0;
        }

        return targets;
    }

    private static double evaluateAccuracy(
            NeuralNetwork network,
            double[][] images,
            int[] labels) {

        if (images.length != labels.length || images.length == 0) {
            throw new IllegalArgumentException(
                "Invalid evaluation dataset");
        }

        int correct = 0;

        for (int i = 0; i < images.length; i++) {
            int predicted = argmax(network.predict(images[i]));

            if (predicted == labels[i]) {
                correct++;
            }
        }

        return (double) correct / images.length;
    }

    private static void evaluate(
            NeuralNetwork network,
            double[][] images,
            int[] labels) {

        int[][] confusion = new int[CLASSES][CLASSES];
        int correct = 0;

        for (int i = 0; i < images.length; i++) {
            int predicted = argmax(network.predict(images[i]));
            int actual = labels[i];

            confusion[actual][predicted]++;

            if (predicted == actual) {
                correct++;
            }
        }

        System.out.printf(
            "Test accuracy: %.2f%% (%d/%d)%n",
            100.0 * correct / images.length,
            correct,
            images.length
        );

        System.out.println("\nConfusion matrix:");
        System.out.println("Rows = actual digits; columns = predictions");

        System.out.print("     ");

        for (int digit = 0; digit < CLASSES; digit++) {
            System.out.printf("%6d", digit);
        }

        System.out.println();

        for (int actual = 0; actual < CLASSES; actual++) {
            System.out.printf("%3d: ", actual);

            for (int predicted = 0; predicted < CLASSES; predicted++) {
                System.out.printf("%6d", confusion[actual][predicted]);
            }

            System.out.println();
        }
    }

    private static int argmax(double[] values) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException(
                "Output must not be empty");
        }

        int best = 0;

        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }

        return best;
    }
}