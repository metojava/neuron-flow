package layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import data.model.Tensor;
import java.util.Random;
import optim.Adam;
import org.junit.jupiter.api.Test;

class MultiHeadAttentionTest {

    private static final int D = 4;
    private static final int H = 2;
    private static final int T = 3;

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

    private static double loss(MultiHeadAttention m, float[][] x, float[][] g) {
        float[][] out = m.forward(x);
        double s = 0;
        for (int t = 0; t < out.length; t++) {
            for (int d = 0; d < D; d++) s += (double) g[t][d] * out[t][d];
        }
        return s;
    }

    @Test
    void gradientsMatchFiniteDifferences() {
        MultiHeadAttention m = new MultiHeadAttention(D, H, 3L);
        float[][] x = random(T, D, 4L);
        float[][] g = random(T, D, 5L);
        float eps = 1e-2f;

        m.zeroGrad();
        m.forward(x);
        float[][] dX = m.backward(g);

        int p = m.parameterCount();
        float[] analytic = new float[p];
        m.copyGradientsTo(analytic, 0);
        float[] params = new float[p];
        m.copyParametersTo(params, 0);
        for (int k = 0; k < p; k++) {
            float orig = params[k];
            params[k] = orig + eps;
            m.copyParametersFrom(params, 0);
            double lp = loss(m, x, g);
            params[k] = orig - eps;
            m.copyParametersFrom(params, 0);
            double lm = loss(m, x, g);
            params[k] = orig;
            double numeric = (lp - lm) / (2.0 * eps);
            assertTrue(Math.abs(analytic[k] - numeric) <= 5e-3 + 1e-2 * Math.abs(numeric),
                    "param " + k + ": analytic " + analytic[k] + " vs numeric " + numeric);
        }
        m.copyParametersFrom(params, 0);

        for (int t = 0; t < T; t++) {
            for (int j = 0; j < D; j++) {
                float orig = x[t][j];
                x[t][j] = orig + eps;
                double lp = loss(m, x, g);
                x[t][j] = orig - eps;
                double lm = loss(m, x, g);
                x[t][j] = orig;
                double numeric = (lp - lm) / (2.0 * eps);
                assertTrue(Math.abs(dX[t][j] - numeric) <= 5e-3 + 1e-2 * Math.abs(numeric),
                        "input [" + t + "][" + j + "]: analytic " + dX[t][j] + " vs " + numeric);
            }
        }
    }

    @Test
    void trainsWithAdam() {
        MultiHeadAttention m = new MultiHeadAttention(D, H, 6L);
        float[][] x = random(T, D, 7L);
        float[][] target = random(T, D, 8L);
        Adam adam = new Adam(0.02f);
        double first = -1;
        double last = -1;
        for (int step = 0; step < 300; step++) {
            m.zeroGrad();
            float[][] out = m.forward(x);
            float[][] grad = new float[T][D];
            double l = 0;
            for (int t = 0; t < T; t++) {
                for (int d = 0; d < D; d++) {
                    float diff = out[t][d] - target[t][d];
                    grad[t][d] = diff;
                    l += 0.5 * diff * diff;
                }
            }
            if (step == 0) first = l;
            last = l;
            m.backward(grad);
            m.update(adam, 1);
        }
        assertTrue(last < 0.5 * first, "Adam did not reduce loss: " + first + " -> " + last);
    }

    /** PyTorch nn.MultiheadAttention forward (batch of one, no masks, no dropout). */
    private static float[][] torchMha(float[][] x, float[] inW, float[] inB,
                                      float[] outW, float[] outB) {
        int n = x.length;
        int dh = D / H;
        float[][] q = new float[n][D];
        float[][] k = new float[n][D];
        float[][] v = new float[n][D];
        for (int t = 0; t < n; t++) {
            for (int o = 0; o < D; o++) {
                float sq = inB[o], sk = inB[D + o], sv = inB[2 * D + o];
                for (int j = 0; j < D; j++) {
                    sq += inW[o * D + j] * x[t][j];
                    sk += inW[(D + o) * D + j] * x[t][j];
                    sv += inW[(2 * D + o) * D + j] * x[t][j];
                }
                q[t][o] = sq;
                k[t][o] = sk;
                v[t][o] = sv;
            }
        }
        float[][] concat = new float[n][D];
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < n; i++) {
                double[] s = new double[n];
                double max = -1e30;
                for (int j = 0; j < n; j++) {
                    double dot = 0;
                    for (int c = 0; c < dh; c++) dot += q[i][h * dh + c] * k[j][h * dh + c];
                    s[j] = dot / Math.sqrt(dh);
                    max = Math.max(max, s[j]);
                }
                double sum = 0;
                for (int j = 0; j < n; j++) {
                    s[j] = Math.exp(s[j] - max);
                    sum += s[j];
                }
                for (int c = 0; c < dh; c++) {
                    double acc = 0;
                    for (int j = 0; j < n; j++) acc += s[j] / sum * v[j][h * dh + c];
                    concat[i][h * dh + c] = (float) acc;
                }
            }
        }
        float[][] out = new float[n][D];
        for (int t = 0; t < n; t++) {
            for (int o = 0; o < D; o++) {
                float s = outB[o];
                for (int j = 0; j < D; j++) s += outW[o * D + j] * concat[t][j];
                out[t][o] = s;
            }
        }
        return out;
    }

    @Test
    void importedWeightsMatchPyTorchReference() {
        float[] inW = randomVec(3 * D * D, 11L);
        float[] inB = randomVec(3 * D, 12L);
        float[] outW = randomVec(D * D, 13L);
        float[] outB = randomVec(D, 14L);

        MultiHeadAttention m = new MultiHeadAttention(D, H, 1L);
        WeightImporter.loadMultiHeadAttention(m,
                new Tensor("in_proj_weight", new int[] {3 * D, D}, inW),
                new Tensor("in_proj_bias", new int[] {3 * D}, inB),
                new Tensor("out_proj.weight", new int[] {D, D}, outW),
                new Tensor("out_proj.bias", new int[] {D}, outB));

        float[][] x = random(T, D, 15L);
        float[][] expected = torchMha(x, inW, inB, outW, outB);
        float[][] actual = m.forward(x);
        for (int t = 0; t < T; t++) {
            assertArrayEquals(expected[t], actual[t], 1e-4f);
        }
    }

    @Test
    void rejectsHeadCountThatDoesNotDivideDim() {
        assertThrows(IllegalArgumentException.class, () -> new MultiHeadAttention(5, 2, 1L));
    }
}
