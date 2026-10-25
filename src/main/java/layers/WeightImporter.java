package layers;

import data.model.Tensor;

/**
 * Copies tensors loaded from model files (safetensors, GGUF, ONNX) into
 * NeuronFlow layers.
 *
 * <p>Tensor layouts follow the PyTorch and ONNX conventions:
 * <ul>
 *   <li>Linear / Dense weight: [out, in]. {@link Dense} stores weights as
 *       [in, out], so this method transposes them.</li>
 *   <li>Conv2D weight: [out, in, kH, kW], the same layout as {@link Conv2D}.</li>
 *   <li>Embedding table: [vocab, dim].</li>
 *   <li>LayerNorm gamma and beta: [features].</li>
 * </ul>
 *
 * <p>Each method checks shapes before writing, so a mismatched tensor leaves
 * the layer unchanged. Passing a {@code null} bias sets the bias to zero.
 */
public final class WeightImporter {

    private WeightImporter() {
    }

    /**
     * Loads a fully connected layer.
     *
     * @param layer  destination layer
     * @param weight tensor of shape [out, in]
     * @param bias   tensor of shape [out], or null for zero bias
     * @throws IllegalArgumentException on shape mismatch
     */
    public static void loadDense(Dense layer, Tensor weight, Tensor bias) {
        requireShape(weight, layer.out, layer.in);
        if (bias != null) requireShape(bias, layer.out);

        float[] src = weight.data();
        for (int j = 0; j < layer.out; j++) {
            for (int i = 0; i < layer.in; i++) {
                layer.w[i * layer.out + j] = src[j * layer.in + i];
            }
        }
        if (bias != null) {
            System.arraycopy(bias.data(), 0, layer.b, 0, layer.out);
        } else {
            java.util.Arrays.fill(layer.b, 0.0f);
        }
    }

    /**
     * Loads a 2-D convolution.
     *
     * @param layer  destination layer
     * @param weight tensor of shape [out, in, kH, kW]
     * @param bias   tensor of shape [out], or null for zero bias
     * @throws IllegalArgumentException on shape mismatch
     */
    public static void loadConv2D(Conv2D layer, Tensor weight, Tensor bias) {
        float[][][][] w = layer.weights;
        int outC = w.length;
        int inC = w[0].length;
        int kH = w[0][0].length;
        int kW = w[0][0][0].length;
        requireShape(weight, outC, inC, kH, kW);
        if (bias != null) requireShape(bias, outC);

        float[] src = weight.data();
        int k = 0;
        for (int o = 0; o < outC; o++) {
            for (int c = 0; c < inC; c++) {
                for (int y = 0; y < kH; y++) {
                    for (int x = 0; x < kW; x++) {
                        w[o][c][y][x] = src[k++];
                    }
                }
            }
        }
        if (bias != null) {
            System.arraycopy(bias.data(), 0, layer.biases, 0, outC);
        } else {
            java.util.Arrays.fill(layer.biases, 0.0f);
        }
    }

    /**
     * Loads an embedding table.
     *
     * @param layer destination layer
     * @param table tensor of shape [vocab, dim]
     * @throws IllegalArgumentException on shape mismatch
     */
    public static void loadEmbedding(Embedding layer, Tensor table) {
        requireShape(table, layer.vocabSize(), layer.embeddingDim());
        float[][] rows = new float[layer.vocabSize()][layer.embeddingDim()];
        float[] src = table.data();
        for (int v = 0; v < rows.length; v++) {
            System.arraycopy(src, v * layer.embeddingDim(), rows[v], 0, layer.embeddingDim());
        }
        layer.loadWeights(rows);
    }

    /**
     * Loads LayerNorm scale and shift.
     *
     * @param layer destination layer
     * @param gamma tensor of shape [features]
     * @param beta  tensor of shape [features], or null for zero shift
     * @param features expected feature count
     * @throws IllegalArgumentException on shape mismatch
     */
    public static void loadLayerNorm(LayerNorm layer, Tensor gamma, Tensor beta, int features) {
        requireShape(gamma, features);
        float[] b = new float[features];
        if (beta != null) {
            requireShape(beta, features);
            b = beta.data();
        }
        layer.loadParameters(gamma.data(), b);
    }

    /**
     * Loads a vanilla RNN in PyTorch layout: weight_ih [H, I], weight_hh [H, H],
     * bias_ih [H] and bias_hh [H]. The two biases are summed.
     *
     * @param layer    destination layer
     * @param weightIh input-to-hidden weights
     * @param weightHh hidden-to-hidden weights
     * @param biasIh   input bias, or null
     * @param biasHh   hidden bias, or null
     * @throws IllegalArgumentException on shape mismatch
     */
    public static void loadRnn(RNN layer, Tensor weightIh, Tensor weightHh,
                               Tensor biasIh, Tensor biasHh) {
        int h = layer.hiddenSize();
        int in = layer.inputSize();
        requireShape(weightIh, h, in);
        requireShape(weightHh, h, h);
        if (biasIh != null) requireShape(biasIh, h);
        if (biasHh != null) requireShape(biasHh, h);

        float[] flat = new float[h * in + h * h + h];
        System.arraycopy(weightIh.data(), 0, flat, 0, h * in);
        System.arraycopy(weightHh.data(), 0, flat, h * in, h * h);
        int b = h * in + h * h;
        for (int i = 0; i < h; i++) {
            flat[b + i] = (biasIh == null ? 0f : biasIh.data()[i])
                    + (biasHh == null ? 0f : biasHh.data()[i]);
        }
        layer.copyParametersFrom(flat, 0);
    }

    /**
     * Loads an LSTM in PyTorch layout: weight_ih [4H, I], weight_hh [4H, H],
     * bias_ih [4H], bias_hh [4H]. PyTorch orders gates i, f, g, o; this layer
     * orders them f, i, g, o, so the first two blocks are swapped. The two
     * biases are summed.
     *
     * @param layer    destination layer
     * @param weightIh input-to-gate weights
     * @param weightHh hidden-to-gate weights
     * @param biasIh   input bias, or null
     * @param biasHh   hidden bias, or null
     * @throws IllegalArgumentException on shape mismatch
     */
    public static void loadLstm(LSTM layer, Tensor weightIh, Tensor weightHh,
                                Tensor biasIh, Tensor biasHh) {
        int h = layer.hiddenSize();
        int in = layer.inputSize();
        int total = 4 * h;
        requireShape(weightIh, total, in);
        requireShape(weightHh, total, h);
        if (biasIh != null) requireShape(biasIh, total);
        if (biasHh != null) requireShape(biasHh, total);

        // Our block k (f, i, g, o) comes from PyTorch block PERM[k] (i, f, g, o).
        int[] perm = {1, 0, 2, 3};
        int rowLen = in + h;
        float[] flat = new float[total * rowLen + total];
        float[] wi = weightIh.data();
        float[] wh = weightHh.data();
        for (int r = 0; r < total; r++) {
            int src = perm[r / h] * h + r % h;
            System.arraycopy(wi, src * in, flat, r * rowLen, in);
            System.arraycopy(wh, src * h, flat, r * rowLen + in, h);
        }
        int b = total * rowLen;
        for (int r = 0; r < total; r++) {
            int src = perm[r / h] * h + r % h;
            flat[b + r] = (biasIh == null ? 0f : biasIh.data()[src])
                    + (biasHh == null ? 0f : biasHh.data()[src]);
        }
        layer.copyParametersFrom(flat, 0);
    }

    /**
     * Loads a GRU in PyTorch layout: weight_ih [3H, I], weight_hh [3H, H],
     * bias_ih [3H], bias_hh [3H]. The layer must use the reset-after formulation.
     *
     * <p>PyTorch computes h = (1 - z) * n + z * h_prev. This layer computes
     * h = (1 - z') * h_prev + z' * n, so z' = 1 - z. Since sigmoid(-a) = 1 - sigmoid(a),
     * the update-gate weights and biases are negated on import.
     *
     * @param layer    destination layer, created with resetAfter = true
     * @param weightIh input-to-gate weights
     * @param weightHh hidden-to-gate weights
     * @param biasIh   input bias, or null
     * @param biasHh   hidden bias, or null
     * @throws IllegalArgumentException on shape mismatch or a reset-before layer
     */
    public static void loadGru(GRU layer, Tensor weightIh, Tensor weightHh,
                               Tensor biasIh, Tensor biasHh) {
        if (!layer.isResetAfter()) {
            throw new IllegalArgumentException(
                "loadGru requires a GRU created with resetAfter = true");
        }
        int h = layer.hiddenSize();
        int in = layer.inputSize();
        int total = 3 * h;
        requireShape(weightIh, total, in);
        requireShape(weightHh, total, h);
        if (biasIh != null) requireShape(biasIh, total);
        if (biasHh != null) requireShape(biasHh, total);

        int rowLen = in + h;
        float[] flat = new float[total * rowLen + total + h];
        float[] wi = weightIh.data();
        float[] wh = weightHh.data();
        for (int r = 0; r < total; r++) {
            float sign = (r / h == 1) ? -1f : 1f; // update gate
            for (int j = 0; j < in; j++) flat[r * rowLen + j] = sign * wi[r * in + j];
            for (int j = 0; j < h; j++) flat[r * rowLen + in + j] = sign * wh[r * h + j];
        }
        int b0 = total * rowLen;
        for (int r = 0; r < total; r++) {
            float sign = (r / h == 1) ? -1f : 1f;
            float bi = biasIh == null ? 0f : biasIh.data()[r];
            float bh = biasHh == null ? 0f : biasHh.data()[r];
            // Reset and update gates add both biases. The candidate's input bias
            // goes to b and its hidden bias goes to b_hn, which is applied after the reset gate.
            float value = r < 2 * h ? bi + bh : bi;
            flat[b0 + r] = sign * value;
        }
        for (int i = 0; i < h; i++) {
            flat[b0 + total + i] = biasHh == null ? 0f : biasHh.data()[2 * h + i];
        }
        layer.copyParametersFrom(flat, 0);
    }

    /**
     * Loads single-head self-attention from PyTorch nn.MultiheadAttention
     * (one head): in_proj_weight [3D, D] holds the Q, K, V projections stacked
     * by rows; out_proj.weight [D, D]. The layer has no bias terms, so the
     * biases must be null or all zero.
     *
     * @param layer         destination layer
     * @param inProjWeight  stacked Q, K, V projection weights
     * @param outProjWeight output projection weight
     * @param inProjBias    stacked Q, K, V biases, or null
     * @param outProjBias   output projection bias, or null
     * @throws IllegalArgumentException on shape mismatch or non-zero bias
     */
    public static void loadSelfAttention(SelfAttention layer, Tensor inProjWeight,
                                         Tensor outProjWeight, Tensor inProjBias,
                                         Tensor outProjBias) {
        int d = layer.dim();
        requireShape(inProjWeight, 3 * d, d);
        requireShape(outProjWeight, d, d);
        requireZeroOrNull(inProjBias, 3 * d);
        requireZeroOrNull(outProjBias, d);

        float[][] q = rows(inProjWeight.data(), 0, d, d);
        float[][] k = rows(inProjWeight.data(), d, d, d);
        float[][] v = rows(inProjWeight.data(), 2 * d, d, d);
        float[][] o = rows(outProjWeight.data(), 0, d, d);
        layer.loadWeights(q, k, v, o);
    }

    /**
     * Loads a BatchNorm1d layer: per-feature weight and bias (null means ones
     * and zeros) plus running mean and variance used for inference.
     *
     * @param layer       destination layer
     * @param weight      scale, shape [F], or null
     * @param bias        shift, shape [F], or null
     * @param runningMean running mean, shape [F]
     * @param runningVar  running variance, shape [F]
     * @throws IllegalArgumentException on shape mismatch
     */
    public static void loadBatchNorm(BatchNormalization layer, Tensor weight, Tensor bias,
                                     Tensor runningMean, Tensor runningVar) {
        int f = runningMean.data().length;
        requireShape(runningMean, f);
        requireShape(runningVar, f);
        if (weight != null) requireShape(weight, f);
        if (bias != null) requireShape(bias, f);

        float[] g = new float[f];
        float[] bt = new float[f];
        if (weight == null) java.util.Arrays.fill(g, 1f); else System.arraycopy(weight.data(), 0, g, 0, f);
        if (bias != null) System.arraycopy(bias.data(), 0, bt, 0, f);
        layer.loadParameters(g, bt, runningMean.data(), runningVar.data());
    }

    /**
     * Loads multi-head attention from PyTorch nn.MultiheadAttention:
     * in_proj_weight [3D, D] and in_proj_bias [3D] stack Q, K, V by rows;
     * out_proj.weight [D, D] and out_proj.bias [D]. Biases may be null (zero).
     *
     * @param layer         destination layer
     * @param inProjWeight  stacked Q, K, V projection weights
     * @param inProjBias    stacked Q, K, V biases, or null
     * @param outProjWeight output projection weight
     * @param outProjBias   output projection bias, or null
     * @throws IllegalArgumentException on shape mismatch
     */
    public static void loadMultiHeadAttention(MultiHeadAttention layer, Tensor inProjWeight,
                                              Tensor inProjBias, Tensor outProjWeight,
                                              Tensor outProjBias) {
        int d = layer.dim();
        requireShape(inProjWeight, 3 * d, d);
        requireShape(outProjWeight, d, d);
        if (inProjBias != null) requireShape(inProjBias, 3 * d);
        if (outProjBias != null) requireShape(outProjBias, d);

        float[] ib = inProjBias == null ? new float[3 * d] : inProjBias.data();
        float[] ob = outProjBias == null ? new float[d] : outProjBias.data();
        layer.loadParameters(
                rows(inProjWeight.data(), 0, d, d), slice(ib, 0, d),
                rows(inProjWeight.data(), d, d, d), slice(ib, d, d),
                rows(inProjWeight.data(), 2 * d, d, d), slice(ib, 2 * d, d),
                rows(outProjWeight.data(), 0, d, d), ob);
    }

    private static float[] slice(float[] src, int from, int len) {
        float[] out = new float[len];
        System.arraycopy(src, from, out, 0, len);
        return out;
    }

    private static float[][] rows(float[] flat, int startRow, int count, int width) {
        float[][] out = new float[count][width];
        for (int i = 0; i < count; i++) {
            System.arraycopy(flat, (startRow + i) * width, out[i], 0, width);
        }
        return out;
    }

    private static void requireZeroOrNull(Tensor t, int length) {
        if (t == null) return;
        requireShape(t, length);
        for (float v : t.data()) {
            if (v != 0f) {
                throw new IllegalArgumentException("Tensor " + t.name()
                    + ": SelfAttention has no bias terms; non-zero bias cannot be loaded");
            }
        }
    }

    private static void requireShape(Tensor t, int... expected) {
        int[] s = t.shape();
        boolean ok = s.length == expected.length;
        for (int i = 0; ok && i < s.length; i++) {
            ok = s[i] == expected[i];
        }
        if (!ok) {
            throw new IllegalArgumentException("Tensor " + t.name() + ": expected shape "
                    + java.util.Arrays.toString(expected) + ", got "
                    + java.util.Arrays.toString(s));
        }
    }
}
