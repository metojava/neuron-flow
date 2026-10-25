package evaluation.bootstrap;

import java.util.Random;

/**
 * Bootstrap confidence intervals for a metric. Resamples the data
 * with replacement and computes the metric on each resample.
 */
public final class Bootstrap {

    private Bootstrap() {
    }

    /**
     * Functional interface for a metric computed on (predicted, actual).
     */
    @FunctionalInterface
    public interface Metric {
        double compute(float[] predicted, float[] actual);
    }

    /**
     * Computes a bootstrap confidence interval for a metric.
     *
     * @param predicted  predictions
     * @param actual     ground truth
     * @param metric     metric to compute
     * @param nBootstrap number of bootstrap resamples
     * @param confidence confidence level (e.g. 0.95)
     * @param seed       RNG seed
     * @return {lower, upper} confidence interval bounds
     */
    public static double[] confidenceInterval(float[] predicted, float[] actual,
                                              Metric metric, int nBootstrap,
                                              double confidence, long seed) {
        if (predicted.length != actual.length || predicted.length == 0) {
            throw new IllegalArgumentException("predicted and actual must have same non-zero length");
        }
        Random rng = new Random(seed);
        int n = predicted.length;
        double[] bootstraps = new double[nBootstrap];

        for (int b = 0; b < nBootstrap; b++) {
            float[] p = new float[n];
            float[] a = new float[n];
            for (int i = 0; i < n; i++) {
                int idx = rng.nextInt(n);
                p[i] = predicted[idx];
                a[i] = actual[idx];
            }
            bootstraps[b] = metric.compute(p, a);
        }

        java.util.Arrays.sort(bootstraps);
        double alpha = 1.0 - confidence;
        int lo = (int) Math.floor(alpha / 2.0 * nBootstrap);
        int hi = (int) Math.ceil((1.0 - alpha / 2.0) * nBootstrap) - 1;
        lo = Math.max(0, Math.min(lo, nBootstrap - 1));
        hi = Math.max(0, Math.min(hi, nBootstrap - 1));
        return new double[] { bootstraps[lo], bootstraps[hi] };
    }
}
