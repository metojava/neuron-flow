package layers;

import java.util.Random;

/**
 * Transposed convolution (deconvolution): upsamples a feature map.
 * Each input element contributes to a kernel-sized region in the output.
 */
public final class TransposedConv2D {

    private final int outChannels;
    private final int kernelH;
    private final int kernelW;
    private final int stride;
    private final int padding;
    private final float[][][] weights; // [outChannels][kernelH][kernelW]
    private final float[] biases;

    /**
     * @param outChannels number of output filters
     * @param kernelH     kernel height
     * @param kernelW     kernel width
     * @param stride      upsampling stride
     * @param padding     zero-padding
     * @param seed        RNG seed
     */
    public TransposedConv2D(int outChannels, int kernelH, int kernelW,
                            int stride, int padding, long seed) {
        if (outChannels <= 0 || kernelH <= 0 || kernelW <= 0
            || stride <= 0 || padding < 0) {
            throw new IllegalArgumentException("Invalid parameters");
        }
        this.outChannels = outChannels;
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.stride = stride;
        this.padding = padding;
        this.weights = new float[outChannels][kernelH][kernelW];
        this.biases = new float[outChannels];

        Random rand = new Random(seed);
        float scale = (float) Math.sqrt(2.0 / (kernelH * kernelW));
        for (int oc = 0; oc < outChannels; oc++) {
            for (int i = 0; i < kernelH; i++) {
                for (int j = 0; j < kernelW; j++) {
                    weights[oc][i][j] = (float) (rand.nextGaussian() * scale);
                }
            }
        }
    }

    /** Output size for a given input size. */
    public int outputSize(int inputSize) {
        return (inputSize - 1) * stride + kernelH - 2 * padding;
    }

    /**
     * Forward pass (single-channel input).
     *
     * @param input [height][width] feature map
     * @return [outChannels][outH][outW] upsampled feature maps
     */
    public float[][][] forward(float[][] input) {
        int inH = input.length;
        int inW = input[0].length;
        int outH = outputSize(inH);
        int outW = outputSize(inW);
        float[][][] out = new float[outChannels][outH][outW];

        for (int oc = 0; oc < outChannels; oc++) {
            for (int i = 0; i < outH; i++) {
                for (int j = 0; j < outW; j++) {
                    out[oc][i][j] = biases[oc];
                }
            }
            for (int ih = 0; ih < inH; ih++) {
                for (int iw = 0; iw < inW; iw++) {
                    float val = input[ih][iw];
                    for (int kh = 0; kh < kernelH; kh++) {
                        for (int kw = 0; kw < kernelW; kw++) {
                            int oh = ih * stride + kh - padding;
                            int ow = iw * stride + kw - padding;
                            if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                                out[oc][oh][ow] += val * weights[oc][kh][kw];
                            }
                        }
                    }
                }
            }
        }
        return out;
    }

    public int parameterCount() {
        return outChannels * kernelH * kernelW + outChannels;
    }
}
