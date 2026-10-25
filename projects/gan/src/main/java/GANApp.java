import data.FashionMNISTDownloader;
import data.FashionMNISTLoader;
import optim.Adam;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;

/**
 * Project 12 - Generative adversarial network.
 *
 * A generator maps a small noise vector to a 28x28 image while a
 * discriminator learns to tell real images from generated ones.
 * The two networks are trained against each other: the
 * discriminator minimises binary cross entropy on real and fake
 * images, and the generator is pushed through the discriminator
 * (its weights are frozen for that step) so it learns to fool it.
 *
 * Uses Fashion-MNIST when available and falls back to synthetic
 * shapes otherwise, so the demo always runs.
 */
public class GANApp {

    private static final int LATENT = 16;
    private static final int IMAGE = 784;
    private static final int BATCH = 32;
    private static final int ITERS = 500;
    private static final float LR = 0.0005f;
    private static final float D_LR = 0.0002f;
    private static final float REAL_TARGET = 0.9f;
    private static final float FAKE_TARGET = 0.1f;

    private static final int LINEAR = 0;
    private static final int RELU = 1;
    private static final int LEAKY = 2;
    private static final int SIGMOID = 3;

    public static void main(String[] args) throws IOException {
        float[][] real = loadImages();
        System.out.println("Training GAN on " + real.length
            + " images (" + ITERS + " iterations, batch "
            + BATCH + ") ...");

        MLP generator = new MLP(
            new int[] {LATENT, 128, 256, IMAGE},
            new int[] {LEAKY, RELU, SIGMOID}, 1L);
        MLP discriminator = new MLP(
            new int[] {IMAGE, 128, 1},
            new int[] {LEAKY, LINEAR}, 2L);

        Adam genOpt = new Adam(LR);
        Adam discOpt = new Adam(D_LR);
        Random rand = new Random(7L);

        System.out.println(" iter   D loss   G loss   D(real)   D(fake)");
        for (int iter = 1; iter <= ITERS; iter++) {
            double dSum = 0.0;
            double gSum = 0.0;
            double realScore = 0.0;
            double fakeScore = 0.0;

            discriminator.zeroGrad();
            generator.zeroGrad();

            for (int s = 0; s < BATCH; s++) {
                float[] realX = real[rand.nextInt(real.length)];
                float[] fakeX = generator.forward(sampleNoise(
                    LATENT, rand));

                float pReal = MLP.sigmoid(
                    discriminator.forward(realX)[0]);
                discriminator.backwardFromOutputGrad(
                    new float[] {pReal - REAL_TARGET});
                dSum += -Math.log(clamp(pReal));

                float pFake = MLP.sigmoid(
                    discriminator.forward(fakeX)[0]);
                discriminator.backwardFromOutputGrad(
                    new float[] {pFake - FAKE_TARGET});
                dSum += -Math.log(clamp(1.0f - pFake));

                realScore += pReal;
                fakeScore += pFake;
            }
            float[] dParams = discriminator.parameters();
            discOpt.update(dParams, discriminator.gradients());
            discriminator.setParameters(dParams);

            discriminator.zeroGrad();
            generator.zeroGrad();
            for (int s = 0; s < BATCH; s++) {
                float[] z = sampleNoise(LATENT, rand);
                float[] fakeX = generator.forward(z);
                float pFake = MLP.sigmoid(
                    discriminator.forward(fakeX)[0]);
                float[] dFake = discriminator.backwardFromOutputGrad(
                    new float[] {pFake - 1.0f});
                generator.backward(dFake);
                gSum += -Math.log(clamp(pFake));
            }
            float[] gParams = generator.parameters();
            genOpt.update(gParams, generator.gradients());
            generator.setParameters(gParams);

            if (iter == 1 || iter % 50 == 0) {
                System.out.printf(
                    "%5d   %6.3f   %.4f   %.4f    %.4f%n",
                    iter, dSum / (2 * BATCH),
                    gSum / BATCH, realScore / BATCH,
                    fakeScore / BATCH);
            }
        }

        System.out.println();
        System.out.println("Sample generated digits:");
        printSample(generator.forward(sampleNoise(LATENT,
            new Random(99L))));
    }

    private static float[][] loadImages() throws IOException {
        try {
            Path dir = FashionMNISTDownloader.ensureDownloaded(
                Paths.get("data", "fashion-mnist"));
            FashionMNISTLoader.Dataset ds =
                FashionMNISTLoader.load(dir);
            int n = Math.min(5000, ds.trainImages().length);
            return Arrays.copyOf(ds.trainImages(), n);
        } catch (Exception e) {
            System.out.println("Fashion-MNIST unavailable ("
                + e.getMessage() + "); using synthetic data.");
            return synthetic(512);
        }
    }

    private static float[][] synthetic(int n) {
        Random rand = new Random(11L);
        float[][] data = new float[n][IMAGE];
        for (int k = 0; k < n; k++) {
            int cx = 6 + rand.nextInt(16);
            int cy = 6 + rand.nextInt(16);
            int r = 4 + rand.nextInt(4);
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    int dx = x - cx;
                    int dy = y - cy;
                    float v = dx * dx + dy * dy
                        <= r * r ? 1.0f : 0.0f;
                    data[k][y * 28 + x] = v
                        * (0.7f + 0.3f * rand.nextFloat());
                }
            }
        }
        return data;
    }

    private static float[] sampleNoise(int dim, Random rand) {
        float[] z = new float[dim];
        for (int i = 0; i < dim; i++) {
            z[i] = (float) rand.nextGaussian();
        }
        return z;
    }

    private static double clamp(float p) {
        return Math.max(1e-6, Math.min(1.0 - 1e-6, p));
    }

    private static void printSample(float[] image) {
        for (int y = 0; y < 28; y++) {
            StringBuilder row = new StringBuilder();
            for (int x = 0; x < 28; x++) {
                float v = image[y * 28 + x];
                row.append(v > 0.5f ? '#' : v > 0.2f ? '+' : '.');
            }
            System.out.println(row);
        }
    }
}
