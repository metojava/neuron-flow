package layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import data.model.SafetensorsIO;
import data.model.WeightFile;
import java.nio.file.Path;
import java.util.Random;
import optim.Adam;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RecurrentStackTest {

    @TempDir
    Path tmp;

    private static float[][] random(int rows, int cols, long seed) {
        Random r = new Random(seed);
        float[][] m = new float[rows][cols];
        for (float[] row : m) {
            for (int j = 0; j < cols; j++) row[j] = (float) r.nextGaussian();
        }
        return m;
    }

    private static double loss(RecurrentStack s, float[][] x, float[][] g) {
        float[][] out = s.forward(x);
        double sum = 0;
        for (int t = 0; t < out.length; t++) {
            for (int j = 0; j < out[t].length; j++) sum += (double) g[t][j] * out[t][j];
        }
        return sum;
    }

    @Test
    void outputShapeDoublesWhenBidirectional() {
        RecurrentStack s = new RecurrentStack(RecurrentStack.Type.GRU, 3, 4, 2, true, true, 1L);
        assertEquals(8, s.outputSize());
        float[][] out = s.forward(random(5, 3, 2L));
        assertEquals(5, out.length);
        assertEquals(8, out[0].length);
    }

    @Test
    void singleLayerUnidirectionalMatchesItsCell() {
        RecurrentStack s = new RecurrentStack(RecurrentStack.Type.LSTM, 3, 2, 1, false, false, 4L);
        float[][] x = random(4, 3, 5L);
        float[][] fromStack = s.forward(x);
        float[][] fromCell = ((LSTM) s.cell(0, 0)).forward(x);
        for (int t = 0; t < x.length; t++) assertArrayEquals(fromCell[t], fromStack[t], 0f);
    }

    @Test
    void gradientsMatchFiniteDifferencesForBidirectionalTwoLayerLstm() {
        RecurrentStack s = new RecurrentStack(RecurrentStack.Type.LSTM, 2, 2, 2, true, false, 6L);
        float[][] x = random(3, 2, 7L);
        float[][] g = random(3, 4, 8L);
        float eps = 1e-2f;

        s.zeroGrad();
        s.forward(x);
        float[][] dIn = s.backward(g);

        int p = s.parameterCount();
        float[] analytic = new float[p];
        s.copyGradientsTo(analytic, 0);
        float[] params = new float[p];
        s.copyParametersTo(params, 0);
        for (int k = 0; k < p; k++) {
            float orig = params[k];
            params[k] = orig + eps;
            s.copyParametersFrom(params, 0);
            double lp = loss(s, x, g);
            params[k] = orig - eps;
            s.copyParametersFrom(params, 0);
            double lm = loss(s, x, g);
            params[k] = orig;
            double numeric = (lp - lm) / (2.0 * eps);
            assertTrue(Math.abs(analytic[k] - numeric) <= 5e-3 + 1e-2 * Math.abs(numeric),
                    "param " + k + ": analytic " + analytic[k] + " vs numeric " + numeric);
        }
        s.copyParametersFrom(params, 0);

        for (int t = 0; t < x.length; t++) {
            for (int j = 0; j < x[t].length; j++) {
                float orig = x[t][j];
                x[t][j] = orig + eps;
                double lp = loss(s, x, g);
                x[t][j] = orig - eps;
                double lm = loss(s, x, g);
                x[t][j] = orig;
                double numeric = (lp - lm) / (2.0 * eps);
                assertTrue(Math.abs(dIn[t][j] - numeric) <= 5e-3 + 1e-2 * Math.abs(numeric),
                        "input [" + t + "][" + j + "]: analytic " + dIn[t][j] + " vs " + numeric);
            }
        }
    }

    @Test
    void bidirectionalTwoLayerGruTrainsWithAdam() {
        RecurrentStack s = new RecurrentStack(RecurrentStack.Type.GRU, 3, 3, 2, true, true, 9L);
        float[][] x = random(4, 3, 10L);
        float[][] target = random(4, 6, 11L);
        Adam adam = new Adam(0.02f);
        double first = -1;
        double last = -1;
        for (int step = 0; step < 300; step++) {
            s.zeroGrad();
            float[][] out = s.forward(x);
            float[][] grad = new float[4][6];
            double l = 0;
            for (int t = 0; t < 4; t++) {
                for (int j = 0; j < 6; j++) {
                    float d = out[t][j] - target[t][j];
                    grad[t][j] = d;
                    l += 0.5 * d * d;
                }
            }
            if (step == 0) first = l;
            last = l;
            s.backward(grad);
            s.update(adam, 1);
        }
        assertTrue(last < 0.5 * first, "Adam did not reduce loss: " + first + " -> " + last);
    }

    @Test
    void pytorchNamesRoundTripForEveryCellType() throws Exception {
        for (RecurrentStack.Type type : RecurrentStack.Type.values()) {
            RecurrentStack source = new RecurrentStack(type, 3, 2, 2, true, true, 12L);
            float[] params = new float[source.parameterCount()];
            Random r = new Random(13L);
            for (int i = 0; i < params.length; i++) params[i] = (float) r.nextGaussian() * 0.5f;
            source.copyParametersFrom(params, 0);

            Path file = tmp.resolve(type + ".safetensors");
            SafetensorsIO.write(file, ModelLoader.exportRecurrentStack(source));

            RecurrentStack target = new RecurrentStack(type, 3, 2, 2, true, true, 99L);
            ModelLoader.loadRecurrentStack(target, SafetensorsIO.read(file));

            float[][] x = random(4, 3, 14L);
            float[][] expected = source.forward(x);
            float[][] actual = target.forward(x);
            for (int t = 0; t < x.length; t++) {
                assertArrayEquals(expected[t], actual[t], 1e-5f, type.name());
            }
        }
    }

    @Test
    void exportUsesPyTorchNames() {
        RecurrentStack s = new RecurrentStack(RecurrentStack.Type.LSTM, 3, 2, 2, true, false, 15L);
        WeightFile f = ModelLoader.exportRecurrentStack(s);
        assertTrue(f.tensors().containsKey("weight_ih_l0"));
        assertTrue(f.tensors().containsKey("weight_hh_l1_reverse"));
        assertArrayEquals(new int[] {8, 4}, f.tensors().get("weight_ih_l1").shape());
    }

    private static float sig(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    private static float[][] reverseRows(float[][] m) {
        float[][] out = new float[m.length][];
        for (int t = 0; t < m.length; t++) out[t] = m[m.length - 1 - t];
        return out;
    }

    /** One direction of one layer, computed with PyTorch's nn.RNN / nn.LSTM / nn.GRU formulas. */
    private static float[][] torchDirection(WeightFile f, RecurrentStack.Type type, int layer,
                                            String sfx, float[][] seq) {
        String s = "_l" + layer + sfx;
        float[] wih = f.tensors().get("weight_ih" + s).data();
        float[] whh = f.tensors().get("weight_hh" + s).data();
        float[] bih = f.tensors().get("bias_ih" + s).data();
        float[] bhh = f.tensors().get("bias_hh" + s).data();
        int inDim = seq[0].length;
        int gates = type == RecurrentStack.Type.RNN ? 1 : type == RecurrentStack.Type.LSTM ? 4 : 3;
        int h = bih.length / gates;
        float[] hs = new float[h];
        float[] cs = new float[h];
        float[][] out = new float[seq.length][h];
        for (int t = 0; t < seq.length; t++) {
            float[] gi = new float[gates * h];
            float[] gh = new float[gates * h];
            for (int k = 0; k < gates * h; k++) {
                gi[k] = bih[k];
                gh[k] = bhh[k];
                for (int j = 0; j < inDim; j++) gi[k] += wih[k * inDim + j] * seq[t][j];
                for (int j = 0; j < h; j++) gh[k] += whh[k * h + j] * hs[j];
            }
            float[] next = new float[h];
            switch (type) {
                case RNN:
                    for (int k = 0; k < h; k++) next[k] = (float) Math.tanh(gi[k] + gh[k]);
                    break;
                case LSTM:
                    for (int k = 0; k < h; k++) {
                        float i = sig(gi[k] + gh[k]);
                        float fg = sig(gi[h + k] + gh[h + k]);
                        float g = (float) Math.tanh(gi[2 * h + k] + gh[2 * h + k]);
                        float o = sig(gi[3 * h + k] + gh[3 * h + k]);
                        cs[k] = fg * cs[k] + i * g;
                        next[k] = o * (float) Math.tanh(cs[k]);
                    }
                    break;
                default:
                    for (int k = 0; k < h; k++) {
                        float r = sig(gi[k] + gh[k]);
                        float z = sig(gi[h + k] + gh[h + k]);
                        float n = (float) Math.tanh(gi[2 * h + k] + r * gh[2 * h + k]);
                        next[k] = (1 - z) * n + z * hs[k];
                    }
                    break;
            }
            hs = next;
            out[t] = next.clone();
        }
        return out;
    }

    /** Full stack reference: layers in order, bidirectional outputs reversed and concatenated. */
    private static float[][] torchStack(WeightFile f, RecurrentStack.Type type, int layers,
                                        boolean bidir, float[][] x) {
        float[][] input = x;
        for (int l = 0; l < layers; l++) {
            float[][] fwd = torchDirection(f, type, l, "", input);
            if (!bidir) {
                input = fwd;
                continue;
            }
            float[][] bwd = reverseRows(torchDirection(f, type, l, "_reverse", reverseRows(input)));
            float[][] joined = new float[input.length][fwd[0].length + bwd[0].length];
            for (int t = 0; t < input.length; t++) {
                System.arraycopy(fwd[t], 0, joined[t], 0, fwd[t].length);
                System.arraycopy(bwd[t], 0, joined[t], fwd[t].length, bwd[t].length);
            }
            input = joined;
        }
        return input;
    }

    @Test
    void stackMatchesPyTorchReferenceForEveryCellType() {
        for (RecurrentStack.Type type : RecurrentStack.Type.values()) {
            RecurrentStack s = new RecurrentStack(type, 3, 4, 2, true, true, 31L);
            float[] params = new float[s.parameterCount()];
            Random r = new Random(32L);
            for (int i = 0; i < params.length; i++) params[i] = (float) r.nextGaussian() * 0.5f;
            s.copyParametersFrom(params, 0);

            WeightFile exported = ModelLoader.exportRecurrentStack(s);
            float[][] x = random(5, 3, 33L);
            float[][] expected = torchStack(exported, type, 2, true, x);
            float[][] actual = s.forward(x);
            for (int t = 0; t < x.length; t++) {
                assertArrayEquals(expected[t], actual[t], 1e-4f, type.name() + " step " + t);
            }
        }
    }

    @Test
    void resetBeforeGruCannotBeExported() {
        RecurrentStack s = new RecurrentStack(RecurrentStack.Type.GRU, 2, 2, 1, false, false, 16L);
        assertThrows(IllegalArgumentException.class, () -> ModelLoader.exportRecurrentStack(s));
    }
}
