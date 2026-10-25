package data;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;

/**
 * Reads the uncompressed IDX files produced by
 * {@link MNISTDownloader} (after gzip inflation) into float
 * arrays. Images are scaled from [0, 255] to [0, 1].
 */
public class MNISTLoader {

    /** Magic number that identifies an IDX image file. */
    private static final int IMAGE_MAGIC = 2051;

    /** Magic number that identifies an IDX label file. */
    private static final int LABEL_MAGIC = 2049;

    private MNISTLoader() {
    }

    /**
     * Loads an IDX image file: big-endian header
     * [magic, count, rows, cols] followed by count*rows*cols
     * unsigned bytes, each divided by 255.
     *
     * @param path file path (must be uncompressed IDX)
     * @return images, [count][rows*cols], values in [0, 1]
     * @throws IOException on read or format errors
     */
    public static float[][] loadImages(Path path)
            throws IOException {
        try (InputStream in = open(path)) {
            readInt(in);                       // magic
            int count = readInt(in);
            int rows = readInt(in);
            int cols = readInt(in);
            int size = rows * cols;

            float[][] images = new float[count][size];

            for (int i = 0; i < count; i++) {
                for (int j = 0; j < size; j++) {
                    int pixel = in.read();
                    if (pixel < 0) {
                        throw new IOException(
                            "Unexpected end of image data");
                    }
                    images[i][j] = pixel / 255.0f;
                }
            }

            return images;
        }
    }

    /**
     * Loads an IDX label file: big-endian header
     * [magic, count] followed by count unsigned bytes.
     *
     * @param path file path (must be uncompressed IDX)
     * @return labels, [count], values 0..255
     * @throws IOException on read or format errors
     */
    public static int[] loadLabels(Path path) throws IOException {
        try (InputStream in = open(path)) {
            readInt(in);                       // magic
            int count = readInt(in);
            int[] labels = new int[count];

            for (int i = 0; i < count; i++) {
                int label = in.read();
                if (label < 0) {
                    throw new IOException(
                        "Unexpected end of label data");
                }
                labels[i] = label;
            }

            return labels;
        }
    }

    /**
     * Converts integer labels to one-hot float rows.
     *
     * @param labels     integer class ids, [N]
     * @param numClasses number of distinct classes
     * @return one-hot matrix, [N][numClasses]
     * @throws IllegalArgumentException on out-of-range ids
     */
    public static float[][] oneHot(int[] labels, int numClasses) {
        float[][] result = new float[labels.length][numClasses];

        for (int i = 0; i < labels.length; i++) {
            int label = labels[i];

            if (label < 0 || label >= numClasses) {
                throw new IllegalArgumentException(
                    "Label " + label + " out of range at row " + i);
            }

            result[i][label] = 1.0f;
        }

        return result;
    }

    /**
     * Resolves an IDX file path, preferring the exact name and
     * falling back to the {@code .gz} variant the downloader
     * saves. Returns whichever exists; if neither does, returns
     * the plain path so the caller's error message is clear.
     */
    public static Path resolve(Path directory, String baseName) {
        Path plain = directory.resolve(baseName);

        if (Files.exists(plain)) {
            return plain;
        }

        Path gz = directory.resolve(baseName + ".gz");

        if (Files.exists(gz)) {
            return gz;
        }

        return plain;
    }

    /**
     * Opens a path as a stream, transparently gunzipping it
     * when the file starts with the gzip magic bytes 1f 8b.
     * This lets callers pass either the original .gz downloads
     * or manually decompressed files.
     */
    private static InputStream open(Path path) throws IOException {
        InputStream in =
            new java.io.BufferedInputStream(
                Files.newInputStream(path));

        // Peek two bytes to detect gzip without consuming them.
        in.mark(2);
        int b0 = in.read();
        int b1 = in.read();
        in.reset();

        if (b0 == 0x1f && b1 == 0x8b) {
            return new GZIPInputStream(in);
        }

        return in;
    }

    /** Reads one big-endian 32-bit integer. */
    private static int readInt(InputStream in) throws IOException {
        int b0 = in.read();
        int b1 = in.read();
        int b2 = in.read();
        int b3 = in.read();

        if ((b0 | b1 | b2 | b3) < 0) {
            throw new IOException("Unexpected end of IDX header");
        }

        return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
    }
}
