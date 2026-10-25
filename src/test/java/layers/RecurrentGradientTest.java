package layers;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Checks BPTT gradients of LSTM and GRU against central finite differences
 * for both parameters and inputs.
 */
class RecurrentGradientTest {

    private static final float EPS = 1e-2f;
    private static final int SEQ = 4;
    private static final int IN = 3;
    private static final int HID = 2;

    /** Common view of a recurrent cell used by the gradient checks. */
    private interface Cell {
        float[][] forward(float[][] x);
        float[][] backward(float[][] dOut);
        int parameterCount();
        void copyParametersTo(float[] buf, int off);
        void copyParametersFrom(float[] buf, int off);
        void copyGradientsTo(float[] buf, int off);
        void zeroGrad();
        void update(float lr, int batchSize);
    }

    private static Cell lstmCell(long seed) {
        LSTM l = new LSTM(IN, HID, seed);
        return new Cell() {
            public float[][] forward(float[][] x) { return l.forward(x); }
            public float[][] backward(float[][] d) { return l.backward(d); }
            public int parameterCount() { return l.parameterCount(); }
            public void copyParametersTo(float[] b, int o) { l.copyParametersTo(b, o); }
            public void copyParametersFrom(float[] b, int o) { l.copyParametersFrom(b, o); }
            public void copyGradientsTo(float[] b, int o) { l.copyGradientsTo(b, o); }
            public void zeroGrad() { l.zeroGrad(); }
            public void update(float lr, int bs) { l.update(lr, bs); }
        };
    }

    private static Cell gruCell(long seed) {
        GRU g = new GRU(IN, HID, seed);
        return new Cell() {
            public float[][] forward(float[][] x) { return g.forward(x); }
            public float[][] backward(float[][] d) { return g.backward(d); }
            public int parameterCount() { return g.parameterCount(); }
            public void copyParametersTo(float[] b, int o) { g.copyParametersTo(b, o); }
            public void copyParametersFrom(float[] b, int o) { g.copyParametersFrom(b, o); }
            public void copyGradientsTo(float[] b, int o) { g.copyGradientsTo(b, o); }
            public void zeroGrad() { g.zeroGrad(); }
            public void update(float lr, int bs) { g.update(lr, bs); }
        };
    }

    private static float[][] random(int rows, int cols, long seed) {
        Random r = new Random(seed);
        float[][] m = new float[rows][cols];
        for (float[] row : m) {
            for (int j = 0; j < cols; j++) row[j] = (float) r.nextGaussian();
        }
        return m;
    }

    /** L = sum over outputs of G[t][i] * out[t][i]. */
    private static double loss(Cell cell, float[][] x, float[][] g) {
        float[][] out = cell.forward(x);
        double s = 0.0;
        for (int t = 0; t < out.length; t++) {
            for (int i = 0; i < out[t].length; i++) s += (double) g[t][i] * out[t][i];
        }
        return s;
    }

    private static void assertClose(String what, double expected, double actual) {
        double tol = 5e-3 + 1e-2 * Math.abs(expected);
        assertTrue(Math.abs(expected - actual) <= tol,
                what + ": analytic " + actual + " vs numeric " + expected);
    }

    private static void checkGradients(Cell cell) {
        float[][] x = random(SEQ, IN, 11L);
        float[][] g = random(SEQ, HID, 12L);

        cell.zeroGrad();
        cell.forward(x);
        float[][] dIn = cell.backward(g);

        int p = cell.parameterCount();
        float[] analytic = new float[p];
        cell.copyGradientsTo(analytic, 0);
        float[] params = new float[p];
        cell.copyParametersTo(params, 0);

        for (int k = 0; k < p; k++) {
            float orig = params[k];
            params[k] = orig + EPS;
            cell.copyParametersFrom(params, 0);
            double lp = loss(cell, x, g);
            params[k] = orig - EPS;
            cell.copyParametersFrom(params, 0);
            double lm = loss(cell, x, g);
            params[k] = orig;
            double numeric = (lp - lm) / (2.0 * EPS);
            assertClose("param " + k, numeric, analytic[k]);
        }
        cell.copyParametersFrom(params, 0);

        for (int t = 0; t < SEQ; t++) {
            for (int j = 0; j < IN; j++) {
                float orig = x[t][j];
                x[t][j] = orig + EPS;
                double lp = loss(cell, x, g);
                x[t][j] = orig - EPS;
                double lm = loss(cell, x, g);
                x[t][j] = orig;
                assertClose("input [" + t + "][" + j + "]",
                        (lp - lm) / (2.0 * EPS), dIn[t][j]);
            }
        }
    }

    private static void checkTrainingReducesLoss(Cell cell) {
        float[][] x = random(SEQ, IN, 21L);
        float[][] target = random(SEQ, HID, 22L);
        float first = -1f;
        float last = -1f;
        for (int step = 0; step < 200; step++) {
            cell.zeroGrad();
            float[][] out = cell.forward(x);
            float[][] grad = new float[SEQ][HID];
            double l = 0.0;
            for (int t = 0; t < SEQ; t++) {
                for (int i = 0; i < HID; i++) {
                    float d = out[t][i] - target[t][i];
                    grad[t][i] = d;
                    l += 0.5 * d * d;
                }
            }
            if (step == 0) first = (float) l;
            last = (float) l;
            cell.backward(grad);
            cell.update(0.1f, 1);
        }
        assertTrue(last < 0.5f * first, "loss did not drop: " + first + " -> " + last);
    }

    @Test
    void lstmGradientsMatchFiniteDifferences() {
        checkGradients(lstmCell(1L));
    }

    @Test
    void gruGradientsMatchFiniteDifferences() {
        checkGradients(gruCell(2L));
    }

    @Test
    void lstmTrainingReducesLoss() {
        checkTrainingReducesLoss(lstmCell(3L));
    }

    @Test
    void gruTrainingReducesLoss() {
        checkTrainingReducesLoss(gruCell(4L));
    }
}
