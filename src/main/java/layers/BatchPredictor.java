package layers;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Batch inference: runs predictions on a dataset in parallel.
 */
public final class BatchPredictor {

    private BatchPredictor() {
    }

    /**
     * Predicts classes for all images using the given classifier.
     *
     * @param classifier any object with a predict(float[]) method
     * @param images     array of flattened images
     * @param threads    number of worker threads
     * @return array of predicted class indices
     */
    public static int[] predictAll(Predictable classifier, float[][] images,
                                   int threads) {
        if (classifier == null || images == null) {
            throw new IllegalArgumentException("classifier and images must not be null");
        }
        if (threads < 1) threads = 1;

        int n = images.length;
        int[] result = new int[n];
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            @SuppressWarnings("unchecked")
            Future<?>[] futures = new Future[threads];
            int chunk = (n + threads - 1) / threads;
            for (int t = 0; t < threads; t++) {
                final int from = Math.min(t * chunk, n);
                final int to = Math.min(from + chunk, n);
                futures[t] = pool.submit(() -> {
                    for (int i = from; i < to; i++) {
                        result[i] = classifier.predict(images[i]);
                    }
                });
            }
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    throw new RuntimeException("Prediction failed", e);
                }
            }
        } finally {
            pool.shutdown();
        }
        return result;
    }

    /** Single-threaded convenience. */
    public static int[] predictAll(Predictable classifier, float[][] images) {
        return predictAll(classifier, images, 1);
    }

    /** Interface for any classifier with a predict method. */
    public interface Predictable {
        int predict(float[] image);
    }
}
