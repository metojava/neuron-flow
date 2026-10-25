package activations;

public class Activations {
    private Activations() {
    }

    public static ActivationFunction relu() {
        return ActivationFunction.relu();
    }

    public static ActivationFunction leakyRelu(float alpha) {
        return ActivationFunction.leakyRelu(alpha);
    }

    public static ActivationFunction sigmoid() {
        return ActivationFunction.sigmoid();
    }

    public static ActivationFunction tanh() {
        return ActivationFunction.tanh();
    }

    public static ActivationFunction linear() {
        return ActivationFunction.linear();
    }

    public static ActivationFunction softplus() {
        return ActivationFunction.softplus();
    }

    public static ActivationFunction elu() {
        return ActivationFunction.elu();
    }

    public static ActivationFunction selu() {
        return ActivationFunction.selu();
    }

    public static ActivationFunction gelu() {
        return ActivationFunction.gelu();
    }

    public static ActivationFunction swish() {
        return ActivationFunction.swish();
    }

    public static ActivationFunction silu() {
        return ActivationFunction.silu();
    }

    public static ActivationFunction mish() {
        return ActivationFunction.mish();
    }

    public static ActivationFunction relu6() {
        return ActivationFunction.relu6();
    }

    public static ActivationFunction hardswish() {
        return ActivationFunction.hardswish();
    }

    public static ActivationFunction prelu(float alpha) {
        return ActivationFunction.prelu(alpha);
    }

    public static ActivationFunction exponential() {
        return ActivationFunction.exponential();
    }

    /**
     * ReLU applied element-wise to a vector.
     *
     * @param x input values (not modified)
     * @return new array with max(0, x[i])
     */
    public static float[] relu(float[] x) {
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.max(0.0f, x[i]);
        }
        return out;
    }

    /**
     * ReLU gradient: passes dOut where the pre-activation was
     * positive, zero elsewhere.
     *
     * @param dOut gradient w.r.t. the ReLU output
     * @param pre  pre-activation values from the forward pass
     * @return gradient w.r.t. the pre-activation
     */
    public static float[] reluBackward(float[] dOut, float[] pre) {
        float[] d = new float[dOut.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = pre[i] > 0.0f ? dOut[i] : 0.0f;
        }
        return d;
    }

    /**
     * ReLU applied element-wise to a feature map tensor.
     *
     * @param x [channels][height][width] input (not modified)
     * @return new tensor with max(0, x)
     */
    public static float[][][] relu(float[][][] x) {
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

    /**
     * ReLU applied in place to a feature map tensor. Use only when the
     * original values are not needed; the sign of the result matches
     * the sign of the input, so it can serve as the backward mask.
     *
     * @param x [channels][height][width] tensor, modified in place
     */
    public static void reluInPlace(float[][][] x) {
        for (float[][] plane : x) {
            for (float[] row : plane) {
                for (int j = 0; j < row.length; j++) {
                    if (row[j] < 0.0f) {
                        row[j] = 0.0f;
                    }
                }
            }
        }
    }

    /**
     * ReLU gradient applied in place: zeroes dOut wherever the activated
     * value is not positive.
     *
     * @param dOut      gradient w.r.t. the ReLU output, modified in place
     * @param activated ReLU output (or pre-activation; the sign is the same)
     */
    public static void reluBackwardInPlace(float[][][] dOut,
                                           float[][][] activated) {
        for (int c = 0; c < dOut.length; c++) {
            for (int i = 0; i < dOut[c].length; i++) {
                float[] d = dOut[c][i];
                float[] a = activated[c][i];
                for (int j = 0; j < d.length; j++) {
                    if (a[j] <= 0.0f) {
                        d[j] = 0.0f;
                    }
                }
            }
        }
    }

    /**
     * ReLU applied in place to a vector.
     *
     * @param x vector modified in place
     */
    public static void reluInPlace(float[] x) {
        for (int i = 0; i < x.length; i++) {
            if (x[i] < 0.0f) {
                x[i] = 0.0f;
            }
        }
    }

    /**
     * ReLU gradient applied in place to a vector.
     *
     * @param dOut      gradient w.r.t. the ReLU output, modified in place
     * @param activated ReLU output (or pre-activation; the sign is the same)
     */
    public static void reluBackwardInPlace(float[] dOut, float[] activated) {
        for (int i = 0; i < dOut.length; i++) {
            if (activated[i] <= 0.0f) {
                dOut[i] = 0.0f;
            }
        }
    }

    /**
     * ReLU gradient for a feature map tensor.
     *
     * @param dOut gradient w.r.t. the ReLU output, [C][H][W]
     * @param pre  pre-activation tensor from the forward pass, [C][H][W]
     * @return gradient w.r.t. the pre-activation, [C][H][W]
     */
    public static float[][][] reluBackward(float[][][] dOut,
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
}
