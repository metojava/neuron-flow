package layers;

import java.util.Random;

/**
 * Dilated (atrous) convolution: inserts gaps between kernel elements
 * to increase receptive field without increasing parameters.
 */
public final class DilatedConv2D {

    private final int outChannels;
    private final int kernelH;
    private final int kernelW;
    private final int dilation;
    private final int padding;
    private final float[][][] weights; // [outChannels][kernelH][kernelW]
    private final float[] biases;

    /**
     * @param outChannels number of output filters
     * @param kernelH     kernel height
     * @param kernelW     kernel width
     * @param dilation    dilation rate (1 = standard convolution)
     * @param padding     zero-padding
     * @param seed        RNG seed
     */
    public DilatedConv2D(int outChannels, int kernelH, int kernelW,
                         int dilation, int padding, long seed) {
        if (outChannels <= 0 || kernelH <= 0 || kernelW <= 0
            || dilation < 1 || padding < 0) {
            throw new IllegalArgumentException("Invalid parameters");
        }
        this.outChannels = outChannels;
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.dilation = dilation;
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
        int effectiveKernel = (kernelH - 1) * dilation + 1;
        return (inputSize + 2 * padding - effectiveKernel) / 1 + 1;
    }

    /**
     * Forward pass (single-channel input).
     *
     * @param input [height][width] feature map
     * @return [outChannels][outH][outW] feature maps
     */
    public float[][][] forward(float[][] input) {
        int inH = input.length;
        int inW = input[0].length;
        int outH = outputSize(inH);
        int outW = outputSize(inW);
        float[][][] out = new float[outChannels][outH][outW];

        for (int oc = 0; oc < outChannels; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = biases[oc];
                    for (int kh = 0; kh < kernelH; kh++) {
                        for (int kw = 0; kw < kernelW; kw++) {
                            int ih = oh - padding + kh * dilation;
                            int iw = ow - padding + kw * dilation;
                            if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                sum += input[ih][iw] * weights[oc][kh][kw];
                            }
                        }
                    }
                    out[oc][oh][ow] = sum;
                }
            }
        }
        return out;
    }

    public int parameterCount() {
        return outChannels * kernelH * kernelW + outChannels;
    }
}
