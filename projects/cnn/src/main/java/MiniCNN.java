import java.util.Random;

/**
 * A tiny convolutional network written from scratch for Project
 * 10. The library's Conv2D/MaxPool2D layers are forward-only, so
 * this class supplies its own backward pass:
 *
 * <pre>
 *   input 1x28x28
 *   Conv 8 x 3x3 (pad 1) -&gt; ReLU -&gt; MaxPool 2x2   (8x14x14)
 *   Conv 16 x 3x3 (pad 1) -&gt; ReLU -&gt; MaxPool 2x2  (16x7x7)
 *   Flatten 784 -&gt; Dense 128 -&gt; ReLU -&gt; Dense 10  (logits)
 * </pre>
 *
 * Gradients accumulate over a mini-batch; call
 * {@link #applyUpdate} to step and {@link #zeroGrad} to clear.
 */
public final class MiniCNN {

    private static final int IMAGE = 28;

    private final Conv conv1;
    private final Conv conv2;
    private final Dense dense1;
    private final Dense dense2;
    private final MaxPool pool1 = new MaxPool();
    private final MaxPool pool2 = new MaxPool();

    private float[][][] c1;
    private float[][][] r1;
    private float[][][] p1;
    private float[][][] c2;
    private float[][][] r2;
    private float[][][] p2;
    private float[] flat;
    private float[] h1;
    private float[] a1;
    private float[] logits;

    /**
     * Builds the network with He-initialised weights.
     *
     * @param seed RNG seed for reproducible initialisation
     */
    public MiniCNN(long seed) {
        conv1 = new Conv(1, 8, 3, 1, 1, seed);
        conv2 = new Conv(8, 16, 3, 1, 1, seed + 1);
        dense1 = new Dense(7 * 7 * 16, 128, seed + 2);
        dense2 = new Dense(128, 10, seed + 3);
    }

    /**
     * Runs a forward pass for one flattened 28x28 image and
     * caches every intermediate value needed by
     * {@link #backward}.
     *
     * @param image flattened pixels, length 784
     * @return the 10 output logits
     */
    public float[] forward(float[] image) {
        float[][][] img = reshape(image, 1, IMAGE, IMAGE);
        c1 = conv1.forward(img);
        r1 = relu(c1);
        p1 = pool1.forward(r1);
        c2 = conv2.forward(p1);
        r2 = relu(c2);
        p2 = pool2.forward(r2);
        flat = flatten(p2);
        h1 = dense1.forward(flat);
        a1 = relu(h1);
        logits = dense2.forward(a1);
        return logits;
    }

    /**
     * Backpropagates the softmax cross-entropy gradient for the
     * example cached by the last {@link #forward}, accumulating
     * gradients into every layer.
     *
     * @param label true class index
     * @return the cross-entropy loss for this example
     */
    public float backward(int label) {
        float[] probs = softmax(logits);
        float loss = -(float) Math.log(
            Math.max(probs[label], 1e-12f));

        float[] d = new float[probs.length];
        for (int j = 0; j < d.length; j++) {
            d[j] = probs[j] - (j == label ? 1.0f : 0.0f);
        }

        d = dense2.backward(d);
        d = reluBackward(d, h1);
        d = dense1.backward(d);

        float[][][] dP2 = unflatten(d, 16, 7, 7);
        float[][][] dR2 = pool2.backward(dP2);
        float[][][] dC2 = reluBackward(dR2, c2);

        float[][][] dP1 = conv2.backward(dC2);
        float[][][] dR1 = pool1.backward(dP1);
        float[][][] dC1 = reluBackward(dR1, c1);

        conv1.backward(dC1);
        return loss;
    }

    /** Predicts the class with the highest logit. */
    public int predict(float[] image) {
        float[] out = forward(image);
        int best = 0;
        for (int i = 1; i < out.length; i++) {
            if (out[i] > out[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Applies one gradient step to every layer.
     *
     * @param learningRate step size
     * @param batchSize    examples the gradients were summed over
     */
    public void applyUpdate(float learningRate, int batchSize) {
        conv1.update(learningRate, batchSize);
        conv2.update(learningRate, batchSize);
        dense1.update(learningRate, batchSize);
        dense2.update(learningRate, batchSize);
    }

    /** Clears accumulated gradients on every layer. */
    public void zeroGrad() {
        conv1.zeroGrad();
        conv2.zeroGrad();
        dense1.zeroGrad();
        dense2.zeroGrad();
    }

    /** @return total number of trainable parameters */
    public int parameterCount() {
        return conv1.parameterCount() + conv2.parameterCount()
            + dense1.parameterCount() + dense2.parameterCount();
    }

    // ── helpers ─────────────────────────────────────────────

    private static float[] softmax(float[] logits) {
        float max = logits[0];
        for (float v : logits) {
            if (v > max) {
                max = v;
            }
        }
        float sum = 0.0f;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    private static float[] relu(float[] x) {
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.max(0.0f, x[i]);
        }
        return out;
    }

    private static float[] reluBackward(float[] dOut, float[] pre) {
        float[] d = new float[dOut.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = pre[i] > 0.0f ? dOut[i] : 0.0f;
        }
        return d;
    }

    private static float[][][] relu(float[][][] x) {
        float[][][] out = new float[x.length][][];
        for (int c = 0; c < x.length; c++) {
            out[c] = new float[x[c].length][];
            for (int i = 0; i < x[c].length; i++) {
                out[c][i] = new float[x[c][i].length];
                for (int j = 0; j < x[c][i].length; j++) {
                    out[c][i][j] = Math.max(0.0f, x[c][i][j]);
                }
            }
        }
        return out;
    }

    private static float[][][] reluBackward(float[][][] dOut,
                                            float[][][] pre) {
        float[][][] d = new float[dOut.length][][];
        for (int c = 0; c < dOut.length; c++) {
            d[c] = new float[dOut[c].length][];
            for (int i = 0; i < dOut[c].length; i++) {
                d[c][i] = new float[dOut[c][i].length];
                for (int j = 0; j < dOut[c][i].length; j++) {
                    d[c][i][j] = pre[c][i][j] > 0.0f
                        ? dOut[c][i][j] : 0.0f;
                }
            }
        }
        return d;
    }

    private static float[] flatten(float[][][] tensor) {
        int c = tensor.length;
        int h = tensor[0].length;
        int w = tensor[0][0].length;
        float[] out = new float[c * h * w];
        int k = 0;
        for (int ch = 0; ch < c; ch++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    out[k++] = tensor[ch][i][j];
                }
            }
        }
        return out;
    }

    private static float[][][] unflatten(float[] flat, int c,
                                         int h, int w) {
        float[][][] out = new float[c][h][w];
        int k = 0;
        for (int ch = 0; ch < c; ch++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    out[ch][i][j] = flat[k++];
                }
            }
        }
        return out;
    }

    private static float[][][] reshape(float[] flat, int c,
                                       int h, int w) {
        return unflatten(flat, c, h, w);
    }

    // ── layers ──────────────────────────────────────────────

    private static final class Conv {
        private final int inC;
        private final int outC;
        private final int k;
        private final int stride;
        private final int pad;

        private final float[][][][] w;
        private final float[][][][] dw;
        private final float[] b;
        private final float[] db;

        private float[][][] inCache;
        private int inH;
        private int inW;

        Conv(int inC, int outC, int k, int stride, int pad,
             long seed) {
            this.inC = inC;
            this.outC = outC;
            this.k = k;
            this.stride = stride;
            this.pad = pad;

            this.w = new float[outC][inC][k][k];
            this.dw = new float[outC][inC][k][k];
            this.b = new float[outC];
            this.db = new float[outC];

            Random rand = new Random(seed);
            float std = (float) Math.sqrt(2.0 / (inC * k * k));
            for (int oc = 0; oc < outC; oc++) {
                for (int ic = 0; ic < inC; ic++) {
                    for (int i = 0; i < k; i++) {
                        for (int j = 0; j < k; j++) {
                            w[oc][ic][i][j] = (float)
                                (rand.nextGaussian() * std);
                        }
                    }
                }
            }
        }

        float[][][] forward(float[][][] in) {
            inCache = in;
            inH = in[0].length;
            inW = in[0][0].length;
            int outH = (inH + 2 * pad - k) / stride + 1;
            int outW = (inW + 2 * pad - k) / stride + 1;
            float[][][] out = new float[outC][outH][outW];

            for (int oc = 0; oc < outC; oc++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float sum = b[oc];
                        for (int ic = 0; ic < inC; ic++) {
                            for (int ki = 0; ki < k; ki++) {
                                for (int kj = 0; kj < k; kj++) {
                                    int ih = oh * stride - pad + ki;
                                    int iw = ow * stride - pad + kj;
                                    if (ih >= 0 && ih < inH
                                        && iw >= 0 && iw < inW) {
                                        sum += in[ic][ih][iw]
                                            * w[oc][ic][ki][kj];
                                    }
                                }
                            }
                        }
                        out[oc][oh][ow] = sum;
                    }
                }
            }
            return out;
        }

        float[][][] backward(float[][][] dOut) {
            int outH = dOut[0].length;
            int outW = dOut[0][0].length;
            float[][][] dIn = new float[inC][inH][inW];

            for (int oc = 0; oc < outC; oc++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float g = dOut[oc][oh][ow];
                        db[oc] += g;
                        for (int ic = 0; ic < inC; ic++) {
                            for (int ki = 0; ki < k; ki++) {
                                for (int kj = 0; kj < k; kj++) {
                                    int ih =
                                        oh * stride - pad + ki;
                                    int iw = ow * stride - pad + kj;
                                    if (ih >= 0 && ih < inH
                                        && iw >= 0 && iw < inW) {
                                        dw[oc][ic][ki][kj] += g
                                            * inCache[ic][ih][iw];
                                        dIn[ic][ih][iw] += g
                                            * w[oc][ic][ki][kj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return dIn;
        }

        void update(float learningRate, int batchSize) {
            float scale = learningRate / batchSize;
            for (int oc = 0; oc < outC; oc++) {
                b[oc] -= scale * db[oc];
                for (int ic = 0; ic < inC; ic++) {
                    for (int i = 0; i < k; i++) {
                        for (int j = 0; j < k; j++) {
                            w[oc][ic][i][j] -=
                                scale * dw[oc][ic][i][j];
                        }
                    }
                }
            }
        }

        void zeroGrad() {
            for (int oc = 0; oc < outC; oc++) {
                db[oc] = 0.0f;
                for (int ic = 0; ic < inC; ic++) {
                    for (int i = 0; i < k; i++) {
                        for (int j = 0; j < k; j++) {
                            dw[oc][ic][i][j] = 0.0f;
                        }
                    }
                }
            }
        }

        int parameterCount() {
            return outC * inC * k * k + outC;
        }
    }

    private static final class MaxPool {
        private final int pool = 2;
        private final int stride = 2;

        private int inH;
        private int inW;
        private int[][][] argmax;

        float[][][] forward(float[][][] in) {
            int channels = in.length;
            inH = in[0].length;
            inW = in[0][0].length;
            int outH = (inH - pool) / stride + 1;
            int outW = (inW - pool) / stride + 1;
            float[][][] out = new float[channels][outH][outW];
            argmax = new int[channels][outH][outW];

            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float max = Float.NEGATIVE_INFINITY;
                        int best = 0;
                        for (int ph = 0; ph < pool; ph++) {
                            for (int pw = 0; pw < pool; pw++) {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;
                                if (ih < inH && iw < inW
                                    && in[c][ih][iw] > max) {
                                    max = in[c][ih][iw];
                                    best = ph * pool + pw;
                                }
                            }
                        }
                        out[c][oh][ow] = max;
                        argmax[c][oh][ow] = best;
                    }
                }
            }
            return out;
        }

        float[][][] backward(float[][][] dOut) {
            int channels = dOut.length;
            int outH = dOut[0].length;
            int outW = dOut[0][0].length;
            float[][][] dIn = new float[channels][inH][inW];

            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        int best = argmax[c][oh][ow];
                        int ph = best / pool;
                        int pw = best % pool;
                        int ih = oh * stride + ph;
                        int iw = ow * stride + pw;
                        dIn[c][ih][iw] += dOut[c][oh][ow];
                    }
                }
            }
            return dIn;
        }
    }

    private static final class Dense {
        private final int in;
        private final int out;
        private final float[][] w;
        private final float[][] dw;
        private final float[] b;
        private final float[] db;
        private float[] inCache;

        Dense(int in, int out, long seed) {
            this.in = in;
            this.out = out;
            this.w = new float[in][out];
            this.dw = new float[in][out];
            this.b = new float[out];
            this.db = new float[out];

            Random rand = new Random(seed);
            float std = (float) Math.sqrt(2.0 / in);
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    w[i][j] = (float) (rand.nextGaussian() * std);
                }
            }
        }

        float[] forward(float[] x) {
            inCache = x.clone();
            float[] result = new float[out];
            for (int j = 0; j < out; j++) {
                float sum = b[j];
                for (int i = 0; i < in; i++) {
                    sum += x[i] * w[i][j];
                }
                result[j] = sum;
            }
            return result;
        }

        float[] backward(float[] dOut) {
            float[] dIn = new float[in];
            for (int i = 0; i < in; i++) {
                float sum = 0.0f;
                for (int j = 0; j < out; j++) {
                    dw[i][j] += inCache[i] * dOut[j];
                    sum += w[i][j] * dOut[j];
                }
                dIn[i] = sum;
            }
            for (int j = 0; j < out; j++) {
                db[j] += dOut[j];
            }
            return dIn;
        }

        void update(float learningRate, int batchSize) {
            float scale = learningRate / batchSize;
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    w[i][j] -= scale * dw[i][j];
                }
            }
            for (int j = 0; j < out; j++) {
                b[j] -= scale * db[j];
            }
        }

        void zeroGrad() {
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    dw[i][j] = 0.0f;
                }
            }
            for (int j = 0; j < out; j++) {
                db[j] = 0.0f;
            }
        }

        int parameterCount() {
            return in * out + out;
        }
    }
}
