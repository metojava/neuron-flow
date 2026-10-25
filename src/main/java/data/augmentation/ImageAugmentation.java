package data.augmentation;

import java.util.Random;

/**
 * Image augmentation utilities for 28x28 (or arbitrary square) float images.
 * All methods return new arrays; inputs are not modified.
 */
public final class ImageAugmentation {

    private ImageAugmentation() {
    }

    /** Flips an image horizontally. */
    public static float[] horizontalFlip(float[] image, int side) {
        float[] out = new float[image.length];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                out[r * side + (side - 1 - c)] = image[r * side + c];
            }
        }
        return out;
    }

    /** Flips an image vertically. */
    public static float[] verticalFlip(float[] image, int side) {
        float[] out = new float[image.length];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                out[(side - 1 - r) * side + c] = image[r * side + c];
            }
        }
        return out;
    }

    /** Adds Gaussian noise with the given standard deviation, clamped to [0, 1]. */
    public static float[] addNoise(float[] image, float std, Random rng) {
        float[] out = new float[image.length];
        for (int i = 0; i < image.length; i++) {
            float v = image[i] + (float) rng.nextGaussian() * std;
            out[i] = Math.max(0.0f, Math.min(1.0f, v));
        }
        return out;
    }

    /** Randomly shifts the image by up to shift pixels in each direction, padding with zeros. */
    public static float[] randomShift(float[] image, int side, int shift, Random rng) {
        float[] out = new float[image.length];
        int dr = rng.nextInt(2 * shift + 1) - shift;
        int dc = rng.nextInt(2 * shift + 1) - shift;
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                int sr = r - dr;
                int sc = c - dc;
                if (sr >= 0 && sr < side && sc >= 0 && sc < side) {
                    out[r * side + c] = image[sr * side + sc];
                }
            }
        }
        return out;
    }

    /** Applies a random augmentation from the available transforms. */
    public static float[] randomAugment(float[] image, int side, Random rng) {
        int choice = rng.nextInt(4);
        switch (choice) {
            case 0: return horizontalFlip(image, side);
            case 1: return addNoise(image, 0.05f, rng);
            case 2: return randomShift(image, side, 2, rng);
            default: return image.clone();
        }
    }
}
