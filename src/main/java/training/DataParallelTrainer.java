package training;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Data-parallel mini-batch SGD for {@link TrainableClassifier}.
 *
 * <p>Each worker owns a replica (its own forward/backward caches). For
 * every batch the replicas receive the master's weights, each processes
 * a contiguous shard of the batch, and their gradients are summed in a
 * fixed order into the master, which applies one SGD step. The result
 * is therefore the same as serial mini-batch training, up to
 * floating-point summation order, and deterministic for a fixed
 * thread count.
 */
public final class DataParallelTrainer implements AutoCloseable {

    private final TrainableClassifier master;
    private final TrainableClassifier[] replicas;
    private final ExecutorService pool;

    /**
     * @param master         the model whose weights are trained
     * @param replicaFactory creates models with the same architecture as
     *                       master; called once per worker
     * @param threads        number of worker threads, at least 1
     */
    public DataParallelTrainer(TrainableClassifier master, int threads) {
        if (threads < 1) {
            throw new IllegalArgumentException("threads must be >= 1");
        }
        this.master = master;
        this.replicas = new TrainableClassifier[threads];
        for (int t = 0; t < threads; t++) {
            replicas[t] = master.createSharedReplica();
        }
        this.pool = Executors.newFixedThreadPool(threads, runnable -> {
            Thread thread = new Thread(runnable, "dp-trainer");
            thread.setDaemon(true);
            return thread;
        });
    }

    /**
     * Trains one epoch over {@code order[0..count)} in mini-batches.
     *
     * @param images    flattened images, indexed by sample id
     * @param labels    class label per sample id
     * @param order     sample ids in visiting order (shuffled by caller)
     * @param count     number of entries of order to use
     * @param batchSize examples per SGD step
     * @param lr        learning rate
     * @return mean cross-entropy loss over the epoch
     */
    public double trainEpoch(float[][] images, int[] labels, int[] order,
                             int count, int batchSize, float lr) {
        if (batchSize <= 0 || count <= 0) {
            throw new IllegalArgumentException("batchSize and count must be positive");
        }

        int threads = replicas.length;
        float[] gradSum = new float[master.parameterCount()];
        double totalLoss = 0.0;

        for (int start = 0; start < count; start += batchSize) {
            int end = Math.min(start + batchSize, count);
            int n = end - start;
            int chunk = (n + threads - 1) / threads;

            @SuppressWarnings("unchecked")
            Future<Double>[] futures = new Future[threads];
            for (int t = 0; t < threads; t++) {
                final TrainableClassifier replica = replicas[t];
                final int from = Math.min(start + t * chunk, end);
                final int to = Math.min(from + chunk, end);
                futures[t] = pool.submit(() -> {
                    replica.zeroGrad();
                    double loss = 0.0;
                    for (int i = from; i < to; i++) {
                        int id = order[i];
                        replica.forward(images[id]);
                        loss += replica.backward(labels[id]);
                    }
                    return loss;
                });
            }

            Arrays.fill(gradSum, 0.0f);
            for (int t = 0; t < threads; t++) {
                totalLoss += await(futures[t]);
                float[] g = replicas[t].getGradients();
                for (int k = 0; k < g.length; k++) {
                    gradSum[k] += g[k];
                }
            }

            master.setGradients(gradSum);
            master.applyUpdate(lr, n);
        }

        return totalLoss / count;
    }

    private static double await(Future<Double> future) {
        try {
            return future.get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Training interrupted", e);
        } catch (ExecutionException e) {
            throw new IllegalStateException("Worker failed", e.getCause());
        }
    }

    /** Stops the worker threads. The trainer cannot be used afterwards. */
    @Override
    public void close() {
        pool.shutdownNow();
    }
}
