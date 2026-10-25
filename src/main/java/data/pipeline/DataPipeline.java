package data.pipeline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * Simple data pipeline: shuffle, batch, and map operations on
 * float[][] datasets. All operations return new arrays.
 */
public final class DataPipeline {

    private DataPipeline() {
    }

    /**
     * Shuffles a dataset (features + labels) in place using the given RNG.
     *
     * @param features [samples][...] feature arrays
     * @param labels   [samples] labels
     * @param rng      random number generator
     */
    public static void shuffle(float[][] features, int[] labels, Random rng) {
        for (int i = features.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            float[] tmpF = features[i];
            features[i] = features[j];
            features[j] = tmpF;
            int tmpL = labels[i];
            labels[i] = labels[j];
            labels[j] = tmpL;
        }
    }

    /**
     * Creates mini-batch indices for a dataset.
     *
     * @param nSamples  number of samples
     * @param batchSize batch size
     * @param shuffle   whether to shuffle indices
     * @param seed      RNG seed (ignored if shuffle is false)
     * @return list of int[] batches, each containing sample indices
     */
    public static List<int[]> batches(int nSamples, int batchSize,
                                       boolean shuffle, long seed) {
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) indices[i] = i;
        if (shuffle) {
            Random rng = new Random(seed);
            for (int i = nSamples - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }

        List<int[]> result = new ArrayList<>();
        for (int start = 0; start < nSamples; start += batchSize) {
            int end = Math.min(start + batchSize, nSamples);
            int[] batch = new int[end - start];
            System.arraycopy(indices, start, batch, 0, batch.length);
            result.add(batch);
        }
        return result;
    }

    /**
     * Applies a transform to every sample in a dataset.
     *
     * @param data      [samples][features]
     * @param transform function applied to each sample
     * @return new dataset with transformed samples
     */
    public static float[][] map(float[][] data,
                                Function<float[], float[]> transform) {
        float[][] result = new float[data.length][];
        for (int i = 0; i < data.length; i++) {
            result[i] = transform.apply(data[i]);
        }
        return result;
    }
}
