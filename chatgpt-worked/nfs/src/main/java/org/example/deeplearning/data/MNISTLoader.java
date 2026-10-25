package org.example.deeplearning.data;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

public final class MNISTLoader {

    public static final int IMAGE_ROWS = 28;
    public static final int IMAGE_COLS = 28;
    public static final int INPUT_SIZE = IMAGE_ROWS * IMAGE_COLS;
    public static final int NUM_CLASSES = 10;

    private static final int IMAGE_MAGIC = 2051;
    private static final int LABEL_MAGIC = 2049;

    private MNISTLoader() {
    }

    public static final class Dataset {
        private final double[][] images;
        private final int[] labels;

        private Dataset(double[][] images, int[] labels) {
            this.images = images;
            this.labels = labels;
        }

        public int size() {
            return images.length;
        }

        public double[][] getImages() {
            return images;
        }

        public double[] getImage(int index) {
            return images[index];
        }

        public int[] getLabels() {
            return labels;
        }

        public int getLabel(int index) {
            return labels[index];
        }

        public double[][] getOneHotTargets() {
            double[][] targets = new double[labels.length][NUM_CLASSES];

            for (int i = 0; i < labels.length; i++) {
                targets[i][labels[i]] = 1.0;
            }

            return targets;
        }

        public double[] getOneHotTarget(int index) {
            double[] target = new double[NUM_CLASSES];
            target[labels[index]] = 1.0;
            return target;
        }
    }

    public static Dataset loadTraining(Path directory)
            throws IOException {
        return load(
            directory.resolve("train-images-idx3-ubyte"),
            directory.resolve("train-labels-idx1-ubyte"));
    }

    public static Dataset loadTest(Path directory)
            throws IOException {
        return load(
            directory.resolve("t10k-images-idx3-ubyte"),
            directory.resolve("t10k-labels-idx1-ubyte"));
    }

    public static Dataset load(Path imageFile, Path labelFile)
            throws IOException {

        double[][] images = loadImages(imageFile);
        int[] labels = loadLabels(labelFile);

        if (images.length != labels.length) {
            throw new IOException(
                "Image count (" + images.length
                    + ") does not match label count (" + labels.length + ")");
        }

        return new Dataset(images, labels);
    }

    public static double[][] loadImages(Path path) throws IOException {
        try (DataInputStream in = input(path)) {
            int magic = in.readInt();
            int count = in.readInt();
            int rows = in.readInt();
            int cols = in.readInt();

            if (magic != IMAGE_MAGIC) {
                throw new IOException(
                    "Invalid image IDX magic number in " + path);
            }

            if (count <= 0 || rows != IMAGE_ROWS || cols != IMAGE_COLS) {
                throw new IOException(
                    "Unexpected MNIST image dimensions in " + path);
            }

            long expectedLength = 16L + (long) count * rows * cols;
            if (Files.size(path) != expectedLength) {
                throw new IOException(
                    "Unexpected image file size in " + path);
            }

            double[][] images = new double[count][INPUT_SIZE];

            for (int i = 0; i < count; i++) {
                for (int p = 0; p < INPUT_SIZE; p++) {
                    images[i][p] = in.readUnsignedByte() / 255.0;
                }
            }

            ensureEndOfFile(in, path);
            return images;
        } catch (EOFException e) {
            throw new IOException("Truncated image file: " + path, e);
        }
    }

    public static int[] loadLabels(Path path) throws IOException {
        try (DataInputStream in = input(path)) {
            int magic = in.readInt();
            int count = in.readInt();

            if (magic != LABEL_MAGIC || count <= 0) {
                throw new IOException(
                    "Invalid label IDX header in " + path);
            }

            long expectedLength = 8L + count;
            if (Files.size(path) != expectedLength) {
                throw new IOException(
                    "Unexpected label file size in " + path);
            }

            int[] labels = new int[count];

            for (int i = 0; i < count; i++) {
                int label = in.readUnsignedByte();

                if (label >= NUM_CLASSES) {
                    throw new IOException(
                        "Invalid label " + label + " at index " + i);
                }

                labels[i] = label;
            }

            ensureEndOfFile(in, path);
            return labels;
        } catch (EOFException e) {
            throw new IOException("Truncated label file: " + path, e);
        }
    }

    private static DataInputStream input(Path path) throws IOException {
        InputStream stream = Files.newInputStream(path);
        return new DataInputStream(new BufferedInputStream(stream));
    }

    private static void ensureEndOfFile(DataInputStream in, Path path)
            throws IOException {
        if (in.read() != -1) {
            throw new IOException("Unexpected trailing bytes in " + path);
        }
    }
}