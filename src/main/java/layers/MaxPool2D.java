package layers;

/**
 * 2-D max pooling: slides a window of size poolH x poolW with
 * the given stride and keeps only the maximum value in each
 * window. Reduces spatial dimensions, adds no parameters, and
 * provides a small translation invariance.
 */
public class MaxPool2D {

    private final int poolH;
    private final int poolW;
    private final int stride;

    /** Per-window argmax (flat index in the window) from the last multi-channel forward. */
    private int[][][] argmaxCache;
    private int cachedInH;
    private int cachedInW;

    /**
     * @param poolH window height
     * @param poolW window width
     * @param stride step size (usually = poolH or poolW)
     */
    public MaxPool2D(int poolH, int poolW, int stride) {
        if (poolH <= 0 || poolW <= 0 || stride <= 0) {
            throw new IllegalArgumentException(
                "Invalid MaxPool2D parameters");
        }
        this.poolH = poolH;
        this.poolW = poolW;
        this.stride = stride;
    }

    /** Standard 2x2 pooling with stride 2. */
    public static MaxPool2D standard() {
        return new MaxPool2D(2, 2, 2);
    }

    /**
     * Output height/width for a given input size.
     */
    public int outputSize(int inputSize) {
        return (inputSize - poolH) / stride + 1;
    }

    /**
     * Pools a single-channel map.
     *
     * @param input [height][width] feature map
     * @return pooled map, smaller by roughly the pool factor
     */
    public float[][] forward(float[][] input) {
        int inH = input.length;
        int inW = input[0].length;
        int outH = outputSize(inH);
        int outW = outputSize(inW);

        float[][] output = new float[outH][outW];

        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float max = Float.NEGATIVE_INFINITY;

                for (int ph = 0; ph < poolH; ph++) {
                    for (int pw = 0; pw < poolW; pw++) {
                        int ih = oh * stride + ph;
                        int iw = ow * stride + pw;

                        if (ih < inH && iw < inW &&
                            input[ih][iw] > max) {
                            max = input[ih][iw];
                        }
                    }
                }

                output[oh][ow] = max;
            }
        }

        return output;
    }

    /**
     * Pools a multi-channel tensor (each channel pooled
     * independently).
     *
     * @param input [channels][height][width]
     * @return [channels][outH][outW]
     */
    public float[][][] forward(float[][][] input) {
        int channels = input.length;
        int inH = input[0].length;
        int inW = input[0][0].length;
        int outH = outputSize(inH);
        int outW = outputSize(inW);

        float[][][] output = new float[channels][outH][outW];
        argmaxCache = new int[channels][outH][outW];
        cachedInH = inH;
        cachedInW = inW;

        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float max = Float.NEGATIVE_INFINITY;
                    int best = 0;

                    for (int ph = 0; ph < poolH; ph++) {
                        for (int pw = 0; pw < poolW; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;

                            if (ih < inH && iw < inW
                                && input[c][ih][iw] > max) {
                                max = input[c][ih][iw];
                                best = ph * poolW + pw;
                            }
                        }
                    }

                    output[c][oh][ow] = max;
                    argmaxCache[c][oh][ow] = best;
                }
            }
        }

        return output;
    }

    /**
     * Routes each output gradient to the input position that was
     * the maximum in its window during the last
     * {@link #forward(float[][][])}.
     *
     * @param dOut gradient w.r.t. the pooled output, [channels][outH][outW]
     * @return gradient w.r.t. the input, [channels][H][W]
     * @throws IllegalStateException if forward has not been called
     */
    public float[][][] backward(float[][][] dOut) {
        if (argmaxCache == null) {
            throw new IllegalStateException("Call forward before backward");
        }

        int channels = dOut.length;
        int outH = dOut[0].length;
        int outW = dOut[0][0].length;
        float[][][] dIn = new float[channels][cachedInH][cachedInW];

        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int best = argmaxCache[c][oh][ow];
                    int ih = oh * stride + best / poolW;
                    int iw = ow * stride + best % poolW;
                    dIn[c][ih][iw] += dOut[c][oh][ow];
                }
            }
        }

        return dIn;
    }
}
