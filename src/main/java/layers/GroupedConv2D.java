package layers;

import java.util.Random;

/**
 * Grouped convolution: divides input channels into groups, applies
 * separate convolutions per group, concatenates results. Reduces
 * parameters and computation vs standard convolution.
 */
public final class GroupedConv2D {

    private final int inChannels;
    private final int outChannels;
    private final int groups;
    private final int kernelH;
    private final int kernelW;
    private final int padding;
    private final int inPerGroup;
    private final int outPerGroup;

    /** Weights: [outChannels][inChannels/groups][kernelH][kernelW]. */
    public final float[][][][] weights;
    public final float[] biases;

    /**
     * @param inChannels  number of input channels
     * @param outChannels number of output channels
     * @param groups      number of groups (must divide both channel counts)
     * @param kernelH     kernel height
     * @param kernelW     kernel width
     * @param padding     zero-padding
     * @param seed        RNG seed
     */
    public GroupedConv2D(int inChannels, int outChannels, int groups,
                         int kernelH, int kernelW, int padding, long seed) {
        if (inChannels % groups != 0 || outChannels % groups != 0) {
            throw new IllegalArgumentException(
                "channels must be divisible by groups");
        }
        if (inChannels <= 0 || outChannels <= 0 || kernelH <= 0
            || kernelW <= 0 || groups <= 0 || padding < 0) {
            throw new IllegalArgumentException("Invalid parameters");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.groups = groups;
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.padding = padding;
        this.inPerGroup = inChannels / groups;
        this.outPerGroup = outChannels / groups;
        this.weights = new float[outChannels][inPerGroup][kernelH][kernelW];
        this.biases = new float[outChannels];

        Random rand = new Random(seed);
        float scale = (float) Math.sqrt(2.0 / (inPerGroup * kernelH * kernelW));
        for (int oc = 0; oc < outChannels; oc++) {
            for (int ic = 0; ic < inPerGroup; ic++) {
                for (int i = 0; i < kernelH; i++) {
                    for (int j = 0; j < kernelW; j++) {
                        weights[oc][ic][i][j] = (float) (rand.nextGaussian() * scale);
                    }
                }
            }
        }
    }

    /** Output size for a given input size (stride 1). */
    public int outputSize(int inputSize) {
        return inputSize + 2 * padding - kernelH + 1;
    }

    /**
     * Forward pass (single-channel-per-group input is [inChannels][H][W]).
     *
     * @param input [inChannels][H][W]
     * @return [outChannels][outH][outW]
     */
    public float[][][] forward(float[][][] input) {
        int inH = input[0].length;
        int inW = input[0][0].length;
        int outH = outputSize(inH);
        int outW = outputSize(inW);
        float[][][] out = new float[outChannels][outH][outW];

        for (int g = 0; g < groups; g++) {
            int inStart = g * inPerGroup;
            int outStart = g * outPerGroup;
            for (int oc = 0; oc < outPerGroup; oc++) {
                int outIdx = outStart + oc;
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float sum = biases[outIdx];
                        for (int ic = 0; ic < inPerGroup; ic++) {
                            int inIdx = inStart + ic;
                            for (int kh = 0; kh < kernelH; kh++) {
                                for (int kw = 0; kw < kernelW; kw++) {
                                    int ih = oh - padding + kh;
                                    int iw = ow - padding + kw;
                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                        sum += input[inIdx][ih][iw]
                                            * weights[outIdx][ic][kh][kw];
                                    }
                                }
                            }
                        }
                        out[outIdx][oh][ow] = sum;
                    }
                }
            }
        }
        return out;
    }

    public int parameterCount() {
        return outChannels * inPerGroup * kernelH * kernelW + outChannels;
    }
}
