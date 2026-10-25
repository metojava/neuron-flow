package layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import data.model.Tensor;
import java.util.Random;
import optim.Adam;
import org.junit.jupiter.api.Test;

/**
 * RNN gradient check, optimizer training for all recurrent layers, and
 * weight import checked against a reference implementation using PyTorch's
 * formulas.
 */
class RecurrentLayersTest {

    private static final float EPS = 1e-2f;
    private static final int SEQ = 4;
    private static final int IN = 3;
    private static final int HID = 2;

    /** Common view of a recurrent cell. */
    private interface Cell {
        float[][] forward(float[][] x);
        float[][] backward(float[][] dOut);
        int parameterCount();
        void copyParametersTo(float[] buf, int off);
        void copyParametersFrom(float[] buf, int off);
        void copyGradientsTo(float[] buf, int off);
        void zeroGrad();
        void update(Adam adam, int batchSize);
    }

    private static Cell rnnCell(long seed) {
        RNN c = new RNN(IN, HID, seed);
        return new Cell() {
            public float[][] forward(float[][] x) { return c.forward(x); }
            public float[][] backward(float[][] d) { return c.backward(d); }
            public int parameterCount() { return c.parameterCount(); }
            public void copyParametersTo(float[] b, int o) { c.copyParametersTo(b, o); }
            public void copyParametersFrom(float[] b, int o) { c.copyParametersFrom(b, o); }
            public void copyGradientsTo(float[] b, int o) { c.copyGradientsTo(b, o); }
            public void zeroGrad() { c.zeroGrad(); }
            public void update(Adam a, int bs) { c.update(a, bs); }
        };
    }

    private static Cell lstmCell(long seed) {
        LSTM c = new LSTM(IN, HID, seed);
        return new Cell() {
            public float[][] forward(float[][] x) { return c.forward(x); }
            public float[][] backward(float[][] d) { return c.backward(d); }
            public int parameterCount() { return c.parameterCount(); }
            public void copyParametersTo(float[] b, int o) { c.copyParametersTo(b, o); }
            public void copyParametersFrom(float[] b, int o) { c.copyParametersFrom(b, o); }
            public void copyGradientsTo(float[] b, int o) { c.copyGradientsTo(b, o); }
            public void zeroGrad() { c.zeroGrad(); }
            public void update(Adam a, int bs) { c.update(a, bs); }
        };
    }

    private static Cell gruCell(long seed) {
        GRU c = new GRU(IN, HID, seed);
        return new Cell() {
            public float[][] forward(float[][] x) { return c.forward(x); }
            public float[][] backward(float[][] d) { return c.backward(d); }
            public int parameterCount() { return c.parameterCount(); }
            public void copyParametersTo(float[] b, int o) { c.copyParametersTo(b, o); }
            public void copyParametersFrom(float[] b, int o) { c.copyParametersFrom(b, o); }
            public void copyGradientsTo(float[] b, int o) { c.copyGradientsTo(b, o); }
            public void zeroGrad() { c.zeroGrad(); }
            public void update(Adam a, int bs) { c.update(a, bs); }
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

    private static float[] randomVec(int n, long seed) {
        Random r = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float) r.nextGaussian() * 0.5f;
        return v;
    }

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
            assertClose("param " + k, (lp - lm) / (2.0 * EPS), analytic[k]);
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
                assertClose("input [" + t + "][" + j + "]", (lp - lm) / (2.0 * EPS), dIn[t][j]);
            }
        }
    }

    private static void checkAdamReducesLoss(Cell cell) {
        float[][] x = random(SEQ, IN, 21L);
        float[][] target = random(SEQ, HID, 22L);
        Adam adam = new Adam(0.02f);
        double first = -1;
        double last = -1;
        for (int step = 0; step < 300; step++) {
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
            if (step == 0) first = l;
            last = l;
            cell.backward(grad);
            cell.update(adam, 1);
        }
        assertTrue(last < 0.5 * first, "Adam did not reduce loss: " + first + " -> " + last);
    }

    @Test
    void rnnGradientsMatchFiniteDifferences() {
        checkGradients(rnnCell(5L));
    }

    @Test
    void rnnTrainsWithAdam() {
        checkAdamReducesLoss(rnnCell(6L));
    }

    @Test
    void lstmTrainsWithAdam() {
        checkAdamReducesLoss(lstmCell(7L));
    }

    @Test
    void gruTrainsWithAdam() {
        checkAdamReducesLoss(gruCell(8L));
    }

    // ---- import checks against a PyTorch-formula reference ----

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    /** PyTorch nn.RNN forward (tanh), weights in PyTorch layout. */
    private static float[][] torchRnn(float[][] x, float[] wih, float[] whh, float[] bih, float[] bhh) {
        float[] h = new float[HID];
        float[][] out = new float[x.length][HID];
        for (int t = 0; t < x.length; t++) {
            float[] nh = new float[HID];
            for (int k = 0; k < HID; k++) {
                float s = bih[k] + bhh[k];
                for (int j = 0; j < IN; j++) s += wih[k * IN + j] * x[t][j];
                for (int j = 0; j < HID; j++) s += whh[k * HID + j] * h[j];
                nh[k] = (float) Math.tanh(s);
            }
            h = nh;
            out[t] = h.clone();
        }
        return out;
    }

    /** PyTorch nn.LSTM forward; gate order i, f, g, o. */
    private static float[][] torchLstm(float[][] x, float[] wih, float[] whh, float[] bih, float[] bhh) {
        float[] h = new float[HID];
        float[] c = new float[HID];
        float[][] out = new float[x.length][HID];
        for (int t = 0; t < x.length; t++) {
            float[] pre = new float[4 * HID];
            for (int k = 0; k < 4 * HID; k++) {
                float s = bih[k] + bhh[k];
                for (int j = 0; j < IN; j++) s += wih[k * IN + j] * x[t][j];
                for (int j = 0; j < HID; j++) s += whh[k * HID + j] * h[j];
                pre[k] = s;
            }
            float[] nh = new float[HID];
            float[] nc = new float[HID];
            for (int k = 0; k < HID; k++) {
                float i = sigmoid(pre[k]);
                float f = sigmoid(pre[HID + k]);
                float g = (float) Math.tanh(pre[2 * HID + k]);
                float o = sigmoid(pre[3 * HID + k]);
                nc[k] = f * c[k] + i * g;
                nh[k] = o * (float) Math.tanh(nc[k]);
            }
            h = nh;
            c = nc;
            out[t] = h.clone();
        }
        return out;
    }

    @Test
    void importedRnnMatchesPyTorchFormulas() {
        float[] wih = randomVec(HID * IN, 31L);
        float[] whh = randomVec(HID * HID, 32L);
        float[] bih = randomVec(HID, 33L);
        float[] bhh = randomVec(HID, 34L);
        RNN rnn = new RNN(IN, HID, 1L);
        WeightImporter.loadRnn(rnn,
                new Tensor("w_ih", new int[] {HID, IN}, wih),
                new Tensor("w_hh", new int[] {HID, HID}, whh),
                new Tensor("b_ih", new int[] {HID}, bih),
                new Tensor("b_hh", new int[] {HID}, bhh));

        float[][] x = random(SEQ, IN, 41L);
        float[][] expected = torchRnn(x, wih, whh, bih, bhh);
        float[][] actual = rnn.forward(x);
        for (int t = 0; t < SEQ; t++) {
            assertArrayEquals(expected[t], actual[t], 1e-5f);
        }
    }

    @Test
    void importedLstmMatchesPyTorchFormulas() {
        float[] wih = randomVec(4 * HID * IN, 51L);
        float[] whh = randomVec(4 * HID * HID, 52L);
        float[] bih = randomVec(4 * HID, 53L);
        float[] bhh = randomVec(4 * HID, 54L);
        LSTM lstm = new LSTM(IN, HID, 1L);
        WeightImporter.loadLstm(lstm,
                new Tensor("weight_ih_l0", new int[] {4 * HID, IN}, wih),
                new Tensor("weight_hh_l0", new int[] {4 * HID, HID}, whh),
                new Tensor("bias_ih_l0", new int[] {4 * HID}, bih),
                new Tensor("bias_hh_l0", new int[] {4 * HID}, bhh));

        float[][] x = random(SEQ, IN, 61L);
        float[][] expected = torchLstm(x, wih, whh, bih, bhh);
        float[][] actual = lstm.forward(x);
        for (int t = 0; t < SEQ; t++) {
            assertArrayEquals(expected[t], actual[t], 1e-5f);
        }
    }
}
