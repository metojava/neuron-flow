package examples;

import activations.ActivationFunction;
import data.MNISTLoader;
import evaluation.ClassificationMetrics;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.Adam;
import training.Trainer;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * End-to-end handwritten-digit (MNIST) classification demo.
 * Mirrors {@link FashionMNISTRunner} but expects the original
 * MNIST IDX files already present under {@code data/mnist}.
 * Download them from http://yann.lecun.com/exdb/mnist/ or use
 * the Fashion-MNIST runner for an auto-downloading example.
 */
public class MNISTRunner {

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

    public static void main(String[] args) throws IOException {
        Path dataDir = Paths.get("data", "mnist");

        System.out.println("Loading MNIST dataset from "
            + dataDir.toAbsolutePath() + " ...");

        float[][] trainImages = MNISTLoader.loadImages(
            MNISTLoader.resolve(dataDir,
                "train-images-idx3-ubyte"));
        int[] trainLabels = MNISTLoader.loadLabels(
            MNISTLoader.resolve(dataDir,
                "train-labels-idx1-ubyte"));
        float[][] testImages = MNISTLoader.loadImages(
            MNISTLoader.resolve(dataDir,
                "t10k-images-idx3-ubyte"));
        int[] testLabels = MNISTLoader.loadLabels(
            MNISTLoader.resolve(dataDir,
                "t10k-labels-idx1-ubyte"));

        System.out.println("Training examples: " + trainImages.length);
        System.out.println("Test examples: " + testImages.length);

        int trainSize = Math.min(TRAIN_LIMIT, trainImages.length);
        float[][] trainInputs = new float[trainSize][];
        float[][] trainTargets =
            MNISTLoader.oneHot(
                java.util.Arrays.copyOf(trainLabels, trainSize), 10);

        for (int i = 0; i < trainSize; i++) {
            trainInputs[i] = trainImages[i];
        }

        NeuralNetwork network = new NeuralNetwork(
            TOPOLOGY, ACTIVATIONS,
            LossFunction.softmaxCrossEntropy(),
            new Adam(LEARNING_RATE), SEED);

        System.out.println();
        System.out.println("Training...");

        network.fit(new Trainer(EPOCHS, BATCH_SIZE, true, SEED),
            trainInputs, trainTargets);

        int correct = 0;
        for (int i = 0; i < testImages.length; i++) {
            int predicted = ClassificationMetrics.argmax(
                network.predict(testImages[i]));
            if (predicted == testLabels[i]) {
                correct++;
            }
        }

        float acc = (float) correct / testImages.length;
        System.out.printf("Test accuracy: %.2f%% (%d/%d)%n",
            acc * 100, correct, testImages.length);
    }
}
