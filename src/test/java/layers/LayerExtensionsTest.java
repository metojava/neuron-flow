package layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import data.model.Tensor;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Reset-after GRU (gradients and PyTorch import), SelfAttention import, and
 * BatchNorm import.
 */
class LayerExtensionsTest {

    private static final float EPS = 1e-2f;
    private static final int SEQ = 4;
    private static final int IN = 3;
    private static final int HID = 2;

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

    private static double loss(GRU g, float[][] x, float[][] w) {
        float[][] out = g.forward(x);
        double s = 0;
        for (int t = 0; t < out.length; t++) {
            for (int i = 0; i < out[t].length; i++) s += (double) w[t][i] * out[t][i];
        }
        return s;
    }

    @Test
    void resetAfterGruGradientsMatchFiniteDifferences() {
        GRU g = new GRU(IN, HID, true, 9L);
        float[][] x = random(SEQ, IN, 71L);
        float[][] w = random(SEQ, HID, 72L);

        g.zeroGrad();
        g.forward(x);
        float[][] dIn = g.backward(w);

        int p = g.parameterCount();
        float[] analytic = new float[p];
        g.copyGradientsTo(analytic, 0);
        float[] params = new float[p];
        g.copyParametersTo(params, 0);
        for (int k = 0; k < p; k++) {
            float orig = params[k];
            params[k] = orig + EPS;
            g.copyParametersFrom(params, 0);
            double lp = loss(g, x, w);
            params[k] = orig - EPS;
            g.copyParametersFrom(params, 0);
            double lm = loss(g, x, w);
            params[k] = orig;
            double numeric = (lp - lm) / (2.0 * EPS);
            assertTrue(Math.abs(analytic[k] - numeric) <= 5e-3 + 1e-2 * Math.abs(numeric),
                    "param " + k + ": analytic " + analytic[k] + " vs numeric " + numeric);
        }
        g.copyParametersFrom(params, 0);

        for (int t = 0; t < SEQ; t++) {
            for (int j = 0; j < IN; j++) {
                float orig = x[t][j];
                x[t][j] = orig + EPS;
                double lp = loss(g, x, w);
                x[t][j] = orig - EPS;
                double lm = loss(g, x, w);
                x[t][j] = orig;
                double numeric = (lp - lm) / (2.0 * EPS);
                assertTrue(Math.abs(dIn[t][j] - numeric) <= 5e-3 + 1e-2 * Math.abs(numeric),
                        "input [" + t + "][" + j + "]: analytic " + dIn[t][j]
                        + " vs numeric " + numeric);
            }
        }
    }

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    /** PyTorch nn.GRU forward: h = (1 - z) * n + z * h_prev, n = tanh(gi_n + r * gh_n). */
    private static float[][] torchGru(float[][] x, float[] wih, float[] whh, float[] bih, float[] bhh) {
        float[] h = new float[HID];
        float[][] out = new float[x.length][HID];
        for (int t = 0; t < x.length; t++) {
            float[] gi = new float[3 * HID];
            float[] gh = new float[3 * HID];
            for (int k = 0; k < 3 * HID; k++) {
                gi[k] = bih[k];
                gh[k] = bhh[k];
                for (int j = 0; j < IN; j++) gi[k] += wih[k * IN + j] * x[t][j];
                for (int j = 0; j < HID; j++) gh[k] += whh[k * HID + j] * h[j];
            }
            float[] nh = new float[HID];
            for (int i = 0; i < HID; i++) {
                float r = sigmoid(gi[i] + gh[i]);
                float z = sigmoid(gi[HID + i] + gh[HID + i]);
                float n = (float) Math.tanh(gi[2 * HID + i] + r * gh[2 * HID + i]);
                nh[i] = (1 - z) * n + z * h[i];
            }
            h = nh;
            out[t] = h.clone();
        }
        return out;
    }

    @Test
    void importedResetAfterGruMatchesPyTorchFormulas() {
        float[] wih = randomVec(3 * HID * IN, 81L);
        float[] whh = randomVec(3 * HID * HID, 82L);
        float[] bih = randomVec(3 * HID, 83L);
        float[] bhh = randomVec(3 * HID, 84L);
        GRU gru = new GRU(IN, HID, true, 1L);
        WeightImporter.loadGru(gru,
                new Tensor("weight_ih_l0", new int[] {3 * HID, IN}, wih),
                new Tensor("weight_hh_l0", new int[] {3 * HID, HID}, whh),
                new Tensor("bias_ih_l0", new int[] {3 * HID}, bih),
                new Tensor("bias_hh_l0", new int[] {3 * HID}, bhh));

        float[][] x = random(SEQ, IN, 85L);
        float[][] expected = torchGru(x, wih, whh, bih, bhh);
        float[][] actual = gru.forward(x);
        for (int t = 0; t < SEQ; t++) {
            assertArrayEquals(expected[t], actual[t], 1e-5f);
        }
    }

    @Test
    void loadGruRejectsResetBeforeLayer() {
        GRU gru = new GRU(IN, HID, 1L);
        Tensor w = new Tensor("w", new int[] {3 * HID, IN}, new float[3 * HID * IN]);
        Tensor u = new Tensor("u", new int[] {3 * HID, HID}, new float[3 * HID * HID]);
        assertThrows(IllegalArgumentException.class,
                () -> WeightImporter.loadGru(gru, w, u, null, null));
    }

    /** Torch-style single-head attention: softmax(QK^T / sqrt(D)) V, then out_proj. */
    private static float[][] torchAttention(float[][] x, float[] inProj, float[] outProj, int d) {
        int n = x.length;
        float[][] q = new float[n][d];
        float[][] k = new float[n][d];
        float[][] v = new float[n][d];
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    q[t][i] += inProj[i * d + j] * x[t][j];
                    k[t][i] += inProj[(d + i) * d + j] * x[t][j];
                    v[t][i] += inProj[(2 * d + i) * d + j] * x[t][j];
                }
            }
        }
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            float[] score = new float[n];
            float max = -Float.MAX_VALUE;
            for (int j = 0; j < n; j++) {
                float s = 0;
                for (int c = 0; c < d; c++) s += q[i][c] * k[j][c];
                score[j] = s / (float) Math.sqrt(d);
                max = Math.max(max, score[j]);
            }
            float sum = 0;
            for (int j = 0; j < n; j++) {
                score[j] = (float) Math.exp(score[j] - max);
                sum += score[j];
            }
            float[] att = new float[d];
            for (int j = 0; j < n; j++) {
                for (int c = 0; c < d; c++) att[c] += score[j] / sum * v[j][c];
            }
            for (int o = 0; o < d; o++) {
                for (int c = 0; c < d; c++) out[i][o] += outProj[o * d + c] * att[c];
            }
        }
        return out;
    }

    @Test
    void importedSelfAttentionMatchesReference() {
        int d = 4;
        float[] inProj = randomVec(3 * d * d, 91L);
        float[] outProj = randomVec(d * d, 92L);
        SelfAttention att = new SelfAttention(d, 1L);
        WeightImporter.loadSelfAttention(att,
                new Tensor("in_proj_weight", new int[] {3 * d, d}, inProj),
                new Tensor("out_proj.weight", new int[] {d, d}, outProj),
                null, null);

        float[][] x = random(5, d, 93L);
        float[][] expected = torchAttention(x, inProj, outProj, d);
        float[][] actual = att.forward(x);
        for (int t = 0; t < x.length; t++) {
            assertArrayEquals(expected[t], actual[t], 1e-4f);
        }
    }

    @Test
    void selfAttentionRejectsNonZeroBias() {
        SelfAttention att = new SelfAttention(2, 1L);
        Tensor inW = new Tensor("in", new int[] {6, 2}, new float[12]);
        Tensor outW = new Tensor("out", new int[] {2, 2}, new float[4]);
        Tensor bias = new Tensor("in_b", new int[] {6}, new float[] {0, 0, 0, 0, 0, 1f});
        assertThrows(IllegalArgumentException.class,
                () -> WeightImporter.loadSelfAttention(att, inW, outW, bias, null));
    }

    @Test
    void batchNormImportUsesRunningStatistics() {
        BatchNormalization bn = BatchNormalization.of();
        Tensor w = new Tensor("weight", new int[] {2}, new float[] {2f, 0.5f});
        Tensor b = new Tensor("bias", new int[] {2}, new float[] {1f, -1f});
        Tensor m = new Tensor("running_mean", new int[] {2}, new float[] {3f, 0f});
        Tensor v = new Tensor("running_var", new int[] {2}, new float[] {4f, 1f});
        WeightImporter.loadBatchNorm(bn, w, b, m, v);

        float[] x = {5f, 2f};
        bn.normalizeInference(x);
        // feature 0: 2 * (5 - 3) / sqrt(4 + eps) + 1 = 3 ; feature 1: 0.5 * 2 / sqrt(1 + eps) - 1 = 0
        assertArrayEquals(new float[] {3f, 0f}, x, 1e-3f);
    }

    @Test
    void batchNormRejectsMismatchedStatistics() {
        BatchNormalization bn = BatchNormalization.of();
        Tensor m = new Tensor("m", new int[] {2}, new float[2]);
        Tensor v = new Tensor("v", new int[] {3}, new float[3]);
        assertThrows(IllegalArgumentException.class,
                () -> WeightImporter.loadBatchNorm(bn, null, null, m, v));
    }
}
