import activations.ActivationFunction;
import data.FashionMNISTDownloader;
import data.FashionMNISTLoader;
import initialization.WeightInitializer;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.Adam;
import training.Trainer;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * Project 7 - Image autoencoder.
 *
 * Trains an unsupervised 784-128-32-128-784 network where the
 * input image is also the target: the 32-unit bottleneck must
 * capture enough structure to rebuild each image.
 */
public class AutoencoderApp {

    private static final int TRAIN_LIMIT = 10000;
    private static final int EPOCHS = 8;
    private static final int BATCH = 64;

    public static void main(String[] args) throws IOException {
        Path dir = FashionMNISTDownloader.ensureDownloaded(
            Paths.get("data", "fashion-mnist"));
        FashionMNISTLoader.Dataset ds =
            FashionMNISTLoader.load(dir);

        int n = Math.min(TRAIN_LIMIT, ds.trainImages().length);
        float[][] x = Arrays.copyOf(ds.trainImages(), n);
        float[][] test = Arrays.copyOf(ds.testImages(), 5);

        System.out.println("Training autoencoder on " + n
            + " images ...");

        NeuralNetwork net = new NeuralNetwork(
            new int[] {784, 128, 32, 128, 784},
            new ActivationFunction[] {
                ActivationFunction.linear(),
                ActivationFunction.relu(),
                ActivationFunction.tanh(),
                ActivationFunction.relu(),
                ActivationFunction.sigmoid()
            },
            LossFunction.meanSquaredError(),
            new Adam(0.001f),
            42L,
            WeightInitializer.heNormal());

        System.out.println("epoch   reconstruction MSE");
        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            net.fit(new Trainer(1, BATCH, true, epoch), x, x);
            System.out.printf("%5d   %.4f%n", epoch,
                net.calculateLoss(x, x));
        }

        System.out.println();
        for (int i = 0; i < test.length; i++) {
            float[] rebuilt = net.predict(test[i]);
            System.out.printf("image %d  MSE %.4f%n", i,
                meanSquaredError(test[i], rebuilt));
        }
    }

    private static float meanSquaredError(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum / a.length;
    }
}
