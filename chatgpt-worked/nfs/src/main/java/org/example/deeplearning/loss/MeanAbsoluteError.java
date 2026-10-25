package org.example.deeplearning.loss;

/** Mean absolute error over the output components. */
public final class MeanAbsoluteError implements LossFunction {
    @Override public double loss(double[] p, double[] t) {
        check(p, t); double sum = 0;
        for (int i = 0; i < p.length; i++) sum += Math.abs(p[i] - t[i]);
        return sum / p.length;
    }
    @Override public double[] gradient(double[] p, double[] t) {
        check(p, t); double[] g = new double[p.length];
        for (int i = 0; i < p.length; i++) g[i] = Math.signum(p[i] - t[i]) / p.length;
        return g; // subgradient at zero is chosen to be zero
    }
    private static void check(double[] p, double[] t) {
        if (p == null || t == null || p.length == 0 || p.length != t.length)
            throw new IllegalArgumentException("Predictions and targets must have equal nonzero lengths");
    }
}
