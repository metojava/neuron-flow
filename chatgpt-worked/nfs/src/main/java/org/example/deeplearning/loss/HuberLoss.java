package org.example.deeplearning.loss;

/** Quadratic near zero and linear outside delta, averaged over outputs. */
public final class HuberLoss implements LossFunction {
    private final double delta;
    public HuberLoss(double delta) {
        if (!Double.isFinite(delta) || delta <= 0) throw new IllegalArgumentException("delta must be finite and positive");
        this.delta = delta;
    }
    @Override public double loss(double[] p, double[] t) {
        check(p, t); double sum = 0;
        for (int i = 0; i < p.length; i++) {
            double a = Math.abs(p[i] - t[i]);
            sum += a <= delta ? 0.5 * a * a : delta * (a - 0.5 * delta);
        }
        return sum / p.length;
    }
    @Override public double[] gradient(double[] p, double[] t) {
        check(p, t); double[] g = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            double d = p[i] - t[i];
            g[i] = Math.max(-delta, Math.min(delta, d)) / p.length;
        }
        return g;
    }
    private static void check(double[] p, double[] t) {
        if (p == null || t == null || p.length == 0 || p.length != t.length)
            throw new IllegalArgumentException("Predictions and targets must have equal nonzero lengths");
    }
}
