import activations.ActivationFunction;
import data.MNISTLoader;
import evaluation.ClassificationMetrics;
import initialization.WeightInitializer;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.Adam;
import training.Trainer;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Standalone MNIST (handwritten digits) demo using the
 * neuronflow library. Expects the four IDX files under
 * data/mnist. Run with: mvn -pl projects/mnist exec:exec
 * (after installing the parent library with mvn install).
 */
public class MNISTApp {

    public static void main(String[] args) throws IOException {
        Path dir = Paths.get("data", "mnist");

        float[][] trainImages = MNISTLoader.loadImages(
            MNISTLoader.resolve(dir,
                "train-images-idx3-ubyte"));
        int[] trainLabels = MNISTLoader.loadLabels(
            MNISTLoader.resolve(dir,
                "train-labels-idx1-ubyte"));
        float[][] testImages = MNISTLoader.loadImages(
            MNISTLoader.resolve(dir,
                "t10k-images-idx3-ubyte"));
        int[] testLabels = MNISTLoader.loadLabels(
            MNISTLoader.resolve(dir,
                "t10k-labels-idx1-ubyte"));

        System.out.println("Train: " + trainImages.length
            + "  Test: " + testImages.length);

        int trainSize = Math.min(10000, trainImages.length);
        float[][] x = new float[trainSize][];
        float[][] y = MNISTLoader.oneHot(
            java.util.Arrays.copyOf(trainLabels, trainSize), 10);

        for (int i = 0; i < trainSize; i++) {
            x[i] = trainImages[i];
        }

        NeuralNetwork net = new NeuralNetwork(
            new int[] {784, 128, 10},
            new ActivationFunction[] {
                ActivationFunction.linear(),
                ActivationFunction.relu(),
                ActivationFunction.linear()
            },
            LossFunction.softmaxCrossEntropy(),
            new Adam(0.001f), 123L,
            WeightInitializer.heNormal());

        net.fit(new Trainer(10, 32, true, 123L), x, y);

        int correct = 0;
        for (int i = 0; i < testImages.length; i++) {
            int p = ClassificationMetrics.argmax(
                net.predict(testImages[i]));
            if (p == testLabels[i]) {
                correct++;
            }
        }

        System.out.printf("Test accuracy: %.2f%% (%d/%d)%n",
            100.0 * correct / testLabels.length,
            correct, testLabels.length);
    }
}
