package layers;

import java.util.Random;

/**
 * Depthwise separable convolution: depthwise (per-channel) convolution
 * followed by pointwise (1x1) convolution. Much cheaper than standard
 * convolution.
 */
public final class SeparableConv2D {

    private final int inChannels;
    private final int outChannels;
    private final int kernelH;
    private final int kernelW;
    private final int padding;

    /** Depthwise kernels: [inChannels][kernelH][kernelW]. */
    public final float[][][] depthwise;
    /** Pointwise kernels: [outChannels][inChannels]. */
    public final float[][] pointwise;
    public final float[] bias;

    private float[][][] inputCache;

    /**
     * @param inChannels  number of input channels
     * @param outChannels number of output channels
     * @param kernelH     kernel height
     * @param kernelW     kernel width
     * @param padding     zero-padding
     * @param seed        RNG seed
     */
    public SeparableConv2D(int inChannels, int outChannels, int kernelH,
                           int kernelW, int padding, long seed) {
        if (inChannels <= 0 || outChannels <= 0 || kernelH <= 0 || kernelW <= 0) {
            throw new IllegalArgumentException("Invalid parameters");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.padding = padding;
        this.depthwise = new float[inChannels][kernelH][kernelW];
        this.pointwise = new float[outChannels][inChannels];
        this.bias = new float[outChannels];

        Random rand = new Random(seed);
        float dwScale = (float) Math.sqrt(2.0 / (kernelH * kernelW));
        float pwScale = (float) Math.sqrt(2.0 / inChannels);

        for (int c = 0; c < inChannels; c++) {
            for (int i = 0; i < kernelH; i++) {
                for (int j = 0; j < kernelW; j++) {
                    depthwise[c][i][j] = (float) (rand.nextGaussian() * dwScale);
                }
            }
        }
        for (int oc = 0; oc < outChannels; oc++) {
            for (int ic = 0; ic < inChannels; ic++) {
                pointwise[oc][ic] = (float) (rand.nextGaussian() * pwScale);
            }
        }
    }

    /** Output size for a given input size (stride 1). */
    public int outputSize(int inputSize) {
        return inputSize + 2 * padding - kernelH + 1;
    }

    /**
     * Forward pass: depthwise conv -> pointwise conv.
     *
     * @param input [inChannels][H][W]
     * @return [outChannels][outH][outW]
     */
    public float[][][] forward(float[][][] input) {
        inputCache = input;
        int inH = input[0].length;
        int inW = input[0][0].length;
        int outH = outputSize(inH);
        int outW = outputSize(inW);

        // Depthwise: [inChannels][outH][outW]
        float[][][] dw = new float[inChannels][outH][outW];
        for (int ic = 0; ic < inChannels; ic++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < kernelH; kh++) {
                        for (int kw = 0; kw < kernelW; kw++) {
                            int ih = oh - padding + kh;
                            int iw = ow - padding + kw;
                            if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                sum += input[ic][ih][iw] * depthwise[ic][kh][kw];
                            }
                        }
                    }
                    dw[ic][oh][ow] = sum;
                }
            }
        }

        // Pointwise: [outChannels][outH][outW]
        float[][][] out = new float[outChannels][outH][outW];
        for (int oc = 0; oc < outChannels; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = bias[oc];
                    for (int ic = 0; ic < inChannels; ic++) {
                        sum += dw[ic][oh][ow] * pointwise[oc][ic];
                    }
                    out[oc][oh][ow] = sum;
                }
            }
        }
        return out;
    }

    public int parameterCount() {
        int dwParams = inChannels * kernelH * kernelW;
        int pwParams = outChannels * inChannels + outChannels;
        return dwParams + pwParams;
    }
}
