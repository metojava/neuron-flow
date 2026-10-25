package layers;

/**
 * 2-D average pooling: slides a window of size poolH x poolW with
 * the given stride and keeps the mean value in each window. Reduces
 * spatial dimensions, adds no parameters. The backward pass
 * distributes the gradient evenly over all positions in the window.
 */
public class AvgPool2D {

    private final int poolH;
    private final int poolW;
    private final int stride;

    private int cachedInH;
    private int cachedInW;

    /**
     * @param poolH window height
     * @param poolW window width
     * @param stride step size (usually = poolH or poolW)
     * @throws IllegalArgumentException on non-positive parameters
     */
    public AvgPool2D(int poolH, int poolW, int stride) {
        if (poolH <= 0 || poolW <= 0 || stride <= 0) {
            throw new IllegalArgumentException("Invalid AvgPool2D parameters");
        }
        this.poolH = poolH;
        this.poolW = poolW;
        this.stride = stride;
    }

    /** Standard 2x2 pooling with stride 2. */
    public static AvgPool2D standard() {
        return new AvgPool2D(2, 2, 2);
    }

    /** Output height for a given input height. */
    public int outputSize(int inputSize) {
        return (inputSize - poolH) / stride + 1;
    }

    /** Pools a single-channel map. */
    public float[][] forward(float[][] input) {
        int inH = input.length;
        int inW = input[0].length;
        int outH = outputSize(inH);
        int outW = outputSize(inW);
        float[][] output = new float[outH][outW];
        cachedInH = inH;
        cachedInW = inW;
        float inv = 1.0f / (poolH * poolW);

        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float sum = 0.0f;
                for (int ph = 0; ph < poolH; ph++) {
                    for (int pw = 0; pw < poolW; pw++) {
                        int ih = oh * stride + ph;
                        int iw = ow * stride + pw;
                        if (ih < inH && iw < inW) {
                            sum += input[ih][iw];
                        }
                    }
                }
                output[oh][ow] = sum * inv;
            }
        }
        return output;
    }

    /** Pools a multi-channel tensor. */
    public float[][][] forward(float[][][] input) {
        int channels = input.length;
        int outH = outputSize(input[0].length);
        int outW = outputSize(input[0][0].length);
        float[][][] output = new float[channels][outH][outW];
        for (int c = 0; c < channels; c++) {
            output[c] = forward(input[c]);
        }
        return output;
    }

    /**
     * Distributes each output gradient evenly over all positions in
     * its window.
     */
    public float[][][] backward(float[][][] dOut) {
        int channels = dOut.length;
        int outH = dOut[0].length;
        int outW = dOut[0][0].length;
        float[][][] dIn = new float[channels][cachedInH][cachedInW];
        float inv = 1.0f / (poolH * poolW);

        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float g = dOut[c][oh][ow] * inv;
                    for (int ph = 0; ph < poolH; ph++) {
                        for (int pw = 0; pw < poolW; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            if (ih < cachedInH && iw < cachedInW) {
                                dIn[c][ih][iw] += g;
                            }
                        }
                    }
                }
            }
        }
        return dIn;
    }
}
