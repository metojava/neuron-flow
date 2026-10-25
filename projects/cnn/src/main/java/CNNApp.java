import data.FashionMNISTDownloader;
import data.FashionMNISTLoader;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

/**
 * Project 10 - Convolutional image classifier.
 *
 * Trains {@link MiniCNN} (two conv+pool stages and a dense head,
 * with a hand-written backward pass) on Fashion-MNIST. The
 * dataset is downloaded into data/fashion-mnist on the first run.
 *
 * Optional args: trainLimit (default 4000), epochs (default 2).
 */
public class CNNApp {

    public static void main(String[] args) throws IOException {
        int trainLimit = args.length > 0
            ? Integer.parseInt(args[0]) : 4000;
        int epochs = args.length > 1
            ? Integer.parseInt(args[1]) : 2;
        float learningRate = 0.01f;
        int batchSize = 16;

        Path dir = FashionMNISTDownloader.ensureDownloaded(
            Paths.get("data", "fashion-mnist"));
        FashionMNISTLoader.Dataset ds =
            FashionMNISTLoader.load(dir);

        float[][] trainImages = ds.trainImages();
        int[] trainLabels = ds.trainLabels();
        float[][] testImages = ds.testImages();
        int[] testLabels = ds.testLabels();

        int n = Math.min(trainLimit, trainImages.length);

        MiniCNN net = new MiniCNN(42L);
        System.out.println("Parameters: " + net.parameterCount());
        System.out.println("Training on " + n
            + " images for " + epochs + " epochs (batch "
            + batchSize + ", lr " + learningRate + ") ...");

        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Random rand = new Random(123L);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            shuffle(order, rand);
            double epochLoss = 0.0;

            for (int start = 0; start < n; start += batchSize) {
                int end = Math.min(start + batchSize, n);
                int count = end - start;

                net.zeroGrad();
                for (int i = start; i < end; i++) {
                    int idx = order[i];
                    net.forward(trainImages[idx]);
                    epochLoss += net.backward(trainLabels[idx]);
                }
                net.applyUpdate(learningRate, count);
            }
            System.out.printf("epoch %2d   avg loss %.4f%n",
                epoch, epochLoss / n);
        }

        int eval = Math.min(2000, testImages.length);
        int correct = 0;
        for (int i = 0; i < eval; i++) {
            if (net.predict(testImages[i]) == testLabels[i]) {
                correct++;
            }
        }
        System.out.printf("%nTest accuracy: %.2f%% (%d/%d)%n",
            100.0 * correct / eval, correct, eval);
    }

    private static void shuffle(int[] order, Random rand) {
        for (int i = order.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
    }
}
