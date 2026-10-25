import java.util.Random;

/**
 * A small fully connected network with an explicit forward and
 * backward pass, used by the GAN project so gradients can be
 * chained through one network into another.
 *
 * <p>Each layer is affine followed by an activation. Activations
 * are selected per layer: 0 linear, 1 ReLU, 2 leaky ReLU, 3
 * sigmoid. Parameters and gradients can be exported as flat
 * arrays so the library's {@code optim.Adam} can update them.</p>
 */
public final class MLP {

    private final int[] sizes;
    private final int[] activations;
    private final Dense[] layers;

    private float[][] pre;
    private float[][] act;
    private float[] inputCache;

    /**
     * @param sizes       layer sizes, sizes[0] is the input width
     * @param activations one activation code per non-input layer
     * @param seed        RNG seed for weight initialisation
     */
    public MLP(int[] sizes, int[] activations, long seed) {
        if (activations.length != sizes.length - 1) {
            throw new IllegalArgumentException(
                "need one activation per weight layer");
        }
        this.sizes = sizes.clone();
        this.activations = activations.clone();
        this.layers = new Dense[sizes.length - 1];
        this.pre = new float[sizes.length - 1][];
        this.act = new float[sizes.length - 1][];

        Random rand = new Random(seed);
        for (int l = 0; l < layers.length; l++) {
            layers[l] = new Dense(sizes[l], sizes[l + 1], rand);
        }
    }

    /** Runs a forward pass and caches intermediates. */
    public float[] forward(float[] x) {
        inputCache = x.clone();
        float[] current = x;
        for (int l = 0; l < layers.length; l++) {
            float[] z = layers[l].forward(current);
            pre[l] = z;
            act[l] = apply(activations[l], z);
            current = act[l];
        }
        return current;
    }

    /**
     * Backpropagates from a gradient on the final activation.
     *
     * @param dOut dL/d(activation of the last layer)
     * @return dL/d(input)
     */
    public float[] backward(float[] dOut) {
        float[] d = dOut;
        for (int l = layers.length - 1; l >= 0; l--) {
            d = activationBackward(activations[l], d, pre[l],
                act[l]);
            d = layers[l].backward(d);
        }
        return d;
    }

    /**
     * Backpropagates when the gradient on the last layer's
     * pre-activation is already known (used for sigmoid + BCE).
     *
     * @param dPreLast dL/d(pre-activation of the last layer)
     * @return dL/d(input)
     */
    public float[] backwardFromOutputGrad(float[] dPreLast) {
        float[] d = dPreLast;
        for (int l = layers.length - 1; l >= 0; l--) {
            if (l != layers.length - 1) {
                d = activationBackward(activations[l], d,
                    pre[l], act[l]);
            }
            d = layers[l].backward(d);
        }
        return d;
    }

    /** Clears all accumulated gradients. */
    public void zeroGrad() {
        for (Dense layer : layers) {
            layer.zeroGrad();
        }
    }

    /** @return a flat concatenation of all weights and biases */
    public float[] parameters() {
        int count = 0;
        for (Dense layer : layers) {
            count += layer.w.length + layer.b.length;
        }
        float[] out = new float[count];
        int k = 0;
        for (Dense layer : layers) {
            System.arraycopy(layer.w, 0, out, k, layer.w.length);
            k += layer.w.length;
            System.arraycopy(layer.b, 0, out, k, layer.b.length);
            k += layer.b.length;
        }
        return out;
    }

    /** @return a flat concatenation of all parameter gradients */
    public float[] gradients() {
        int count = 0;
        for (Dense layer : layers) {
            count += layer.gw.length + layer.gb.length;
        }
        float[] out = new float[count];
        int k = 0;
        for (Dense layer : layers) {
            System.arraycopy(layer.gw, 0, out, k, layer.gw.length);
            k += layer.gw.length;
            System.arraycopy(layer.gb, 0, out, k, layer.gb.length);
            k += layer.gb.length;
        }
        return out;
    }

    /** Writes a flat parameter array back into the layers. */
    public void setParameters(float[] params) {
        int k = 0;
        for (Dense layer : layers) {
            System.arraycopy(params, k, layer.w, 0,
                layer.w.length);
            k += layer.w.length;
            System.arraycopy(params, k, layer.b, 0,
                layer.b.length);
            k += layer.b.length;
        }
    }

    /** Scales every accumulated gradient by 1/count. */
    public void scaleGradients(float count) {
        for (Dense layer : layers) {
            for (int i = 0; i < layer.gw.length; i++) {
                layer.gw[i] /= count;
            }
            for (int i = 0; i < layer.gb.length; i++) {
                layer.gb[i] /= count;
            }
        }
    }

    // ── activation helpers ──────────────────────────────────

    private static float[] apply(int code, float[] z) {
        float[] out = new float[z.length];
        for (int i = 0; i < z.length; i++) {
            switch (code) {
                case 1:
                    out[i] = Math.max(0.0f, z[i]);
                    break;
                case 2:
                    out[i] = z[i] >= 0.0f ? z[i] : 0.2f * z[i];
                    break;
                case 3:
                    out[i] = sigmoid(z[i]);
                    break;
                default:
                    out[i] = z[i];
            }
        }
        return out;
    }

    private static float[] activationBackward(int code,
                                              float[] dOut,
                                              float[] pre,
                                              float[] act) {
        float[] d = new float[dOut.length];
        for (int i = 0; i < d.length; i++) {
            switch (code) {
                case 1:
                    d[i] = pre[i] > 0.0f ? dOut[i] : 0.0f;
                    break;
                case 2:
                    d[i] = pre[i] >= 0.0f
                        ? dOut[i] : 0.2f * dOut[i];
                    break;
                case 3:
                    d[i] = act[i] * (1.0f - act[i]) * dOut[i];
                    break;
                default:
                    d[i] = dOut[i];
            }
        }
        return d;
    }

    static float sigmoid(float x) {
        if (x >= 0.0f) {
            return 1.0f / (1.0f + (float) Math.exp(-x));
        }
        float e = (float) Math.exp(x);
        return e / (1.0f + e);
    }

    // ── dense layer ─────────────────────────────────────────

    private static final class Dense {
        private final int in;
        private final int out;
        private final float[] w;
        private final float[] b;
        private final float[] gw;
        private final float[] gb;
        private float[] inCache;

        Dense(int in, int out, Random rand) {
            this.in = in;
            this.out = out;
            this.w = new float[in * out];
            this.b = new float[out];
            this.gw = new float[in * out];
            this.gb = new float[out];

            float std = (float) Math.sqrt(2.0 / in);
            for (int i = 0; i < w.length; i++) {
                w[i] = (float) (rand.nextGaussian() * std);
            }
        }

        float[] forward(float[] x) {
            inCache = x.clone();
            float[] z = new float[out];
            for (int j = 0; j < out; j++) {
                float sum = b[j];
                for (int i = 0; i < in; i++) {
                    sum += x[i] * w[i * out + j];
                }
                z[j] = sum;
            }
            return z;
        }

        float[] backward(float[] dOut) {
            float[] dIn = new float[in];
            for (int i = 0; i < in; i++) {
                float sum = 0.0f;
                for (int j = 0; j < out; j++) {
                    gw[i * out + j] += inCache[i] * dOut[j];
                    sum += w[i * out + j] * dOut[j];
                }
                dIn[i] = sum;
            }
            for (int j = 0; j < out; j++) {
                gb[j] += dOut[j];
            }
            return dIn;
        }

        void zeroGrad() {
            java.util.Arrays.fill(gw, 0.0f);
            java.util.Arrays.fill(gb, 0.0f);
        }
    }
}
