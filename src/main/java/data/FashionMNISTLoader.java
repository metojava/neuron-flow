package data;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Loads the Fashion-MNIST dataset: 60,000 training and 10,000
 * test grayscale 28x28 clothing images across 10 classes.
 * Pixels are scaled to [0, 1]; labels stay as integer ids.
 * Class ids: 0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress,
 * 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot.
 */
public final class FashionMNISTLoader {

    /** Human-readable class names in id order. */
    public static final String[] CLASS_NAMES = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    /** Number of classes. */
    public static final int NUM_CLASSES = 10;

    /** Image side length in pixels. */
    public static final int IMAGE_SIZE = 28;

    /** Flattened pixels per image. */
    public static final int INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE;

    private FashionMNISTLoader() {
    }

    /**
     * Loads the full dataset from a directory containing the
     * four uncompressed (or gzipped) IDX files.
     *
     * @param directory where the files live
     * @return a holder with train/test images and labels
     * @throws IOException on read errors
     */
    public static Dataset load(Path directory) throws IOException {
        float[][] trainImages = MNISTLoader.loadImages(
            MNISTLoader.resolve(directory,
                "train-images-idx3-ubyte"));
        int[] trainLabels = MNISTLoader.loadLabels(
            MNISTLoader.resolve(directory,
                "train-labels-idx1-ubyte"));
        float[][] testImages = MNISTLoader.loadImages(
            MNISTLoader.resolve(directory,
                "t10k-images-idx3-ubyte"));
        int[] testLabels = MNISTLoader.loadLabels(
            MNISTLoader.resolve(directory,
                "t10k-labels-idx1-ubyte"));

        return new Dataset(trainImages, trainLabels,
                           testImages, testLabels);
    }

    /** Immutable holder for the four arrays of a split. */
    public record Dataset(
            float[][] trainImages, int[] trainLabels,
            float[][] testImages,  int[] testLabels) {

        /** @return one-hot training targets, [N][10] */
        public float[][] trainTargetsOneHot() {
            return MNISTLoader.oneHot(trainLabels, NUM_CLASSES);
        }

        /** @return one-hot test targets, [N][10] */
        public float[][] testTargetsOneHot() {
            return MNISTLoader.oneHot(testLabels, NUM_CLASSES);
        }
    }
}
