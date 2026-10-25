package layers;

import java.util.Arrays;
import java.util.Random;

import vector.VectorMath;

/**
 * 2-D convolution layer with trainable kernels. Input is
 * [inChannels][H][W]; output is [outChannels][outH][outW]. Stride
 * and zero-padding are configurable. Weights use He-normal
 * initialization with fan-in inChannels * kernelH * kernelW.
 *
 * <p>Training: {@link #forward} caches its input, {@link #backward}
 * accumulates weight and bias gradients for the cached input, and
 * {@link #update} applies the summed gradients over a mini-batch.
 *
 * <p>For stride 1 the layer zero-pads the input once and computes each
 * kernel tap over the whole output plane with one vector call (SIMD via
 * {@link VectorMath}). Other strides use the general scalar loops.
 */
public class Conv2D {

    /** Kernel weights: [outChannels][inChannels][kernelH][kernelW]. */
    public final float[][][][] weights;

    /** Biases: [outChannels]. */
    public final float[] biases;

    private final float[][][][] weightGrads;
    private final float[] biasGrads;

    private final int inChannels;
    private final int outChannels;
    private final int kernelH;
    private final int kernelW;
    private final int stride;
    private final int padding;

    private float[][][] inputCache;
    /** Zero-padded input planes from the last stride-1 forward pass. */
    private float[][] paddedCache;

    /**
     * Single-channel input convenience constructor.
     *
     * @param outChannels number of output filters
     * @param kernelH     kernel height
     * @param kernelW     kernel width
     * @param stride      step size for the sliding window
     * @param padding     zeros added on each border
     * @param seed        RNG seed for weight init
     */
    public Conv2D(int outChannels, int kernelH, int kernelW,
                  int stride, int padding, long seed) {
        this(1, outChannels, kernelH, kernelW, stride, padding, seed);
    }

    /**
     * Creates a Conv2D layer.
     *
     * @param inChannels  number of input channels
     * @param outChannels number of output filters
     * @param kernelH     kernel height
     * @param kernelW     kernel width
     * @param stride      step size for the sliding window
     * @param padding     zeros added on each border
     * @param seed        RNG seed for weight init
     * @throws IllegalArgumentException on invalid parameters
     */
    public Conv2D(int inChannels, int outChannels, int kernelH,
                  int kernelW, int stride, int padding, long seed) {
        this(inChannels, outChannels, kernelH, kernelW, stride, padding,
             new float[outChannels][inChannels][kernelH][kernelW],
             new float[outChannels], seed);
    }

    /** Shared-weights constructor. */
    public Conv2D(int inChannels, int outChannels, int kernelH,
                  int kernelW, int stride, int padding,
                  float[][][][] sharedWeights, float[] sharedBiases,
                  long seed) {
        if (inChannels <= 0 || outChannels <= 0 || kernelH <= 0
            || kernelW <= 0 || stride <= 0 || padding < 0) {
            throw new IllegalArgumentException(
                "Invalid Conv2D parameters");
        }
        if (sharedWeights == null || sharedWeights.length != outChannels
            || sharedBiases == null || sharedBiases.length != outChannels) {
            throw new IllegalArgumentException("Invalid shared arrays");
        }

        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.stride = stride;
        this.padding = padding;
        this.weights = sharedWeights;
        this.weightGrads = new float[outChannels][inChannels][kernelH][kernelW];
        this.biases = sharedBiases;
        this.biasGrads = new float[outChannels];

        if (seed >= 0) {
            Random rand = new Random(seed);
            float scale = (float) Math.sqrt(
                2.0 / (inChannels * kernelH * kernelW));
            for (int oc = 0; oc < outChannels; oc++) {
                for (int ic = 0; ic < inChannels; ic++) {
                    for (int i = 0; i < kernelH; i++) {
                        for (int j = 0; j < kernelW; j++) {
                            weights[oc][ic][i][j] =
                                (float) (rand.nextGaussian() * scale);
                        }
                    }
                }
            }
        }
    }

    /** Returns a replica that shares this layer's weight and bias arrays
     *  but owns its own gradient buffers. */
    public Conv2D createSharedReplica() {
        return new Conv2D(inChannels, outChannels, kernelH, kernelW,
                          stride, padding, weights, biases, -1L);
    }

    /**
     * Output height for a given input height.
     *
     * @param inputSize input height (or width when the kernel is square)
     * @return output size along one spatial dimension
     */
    public int outputSize(int inputSize) {
        return (inputSize + 2 * padding - kernelH) / stride + 1;
    }

    /**
     * Convolves a single-channel input map.
     *
     * @param input [height][width] feature map
     * @return [outChannels][outH][outW] feature maps
     */
    public float[][][] forward(float[][] input) {
        return forward(new float[][][] {input});
    }

    /**
     * Convolves a multi-channel input and caches it for
     * {@link #backward}.
     *
     * @param input [inChannels][height][width] feature maps
     * @return [outChannels][outH][outW] feature maps
     * @throws IllegalArgumentException if the channel count differs
     */
    public float[][][] forward(float[][][] input) {
        if (input.length != inChannels) {
            throw new IllegalArgumentException(
                "Expected " + inChannels + " input channels, got "
                + input.length);
        }

        inputCache = input;
        int inH = input[0].length;
        int inW = input[0][0].length;
        int outH = outputSize(inH);
        int outW = outputSize(inW);

        if (stride == 1) {
            return forwardStride1(input, inH, inW, outH, outW);
        }

        float[][][] output = new float[outChannels][outH][outW];

        for (int oc = 0; oc < outChannels; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = biases[oc];

                    for (int ic = 0; ic < inChannels; ic++) {
                        for (int kh = 0; kh < kernelH; kh++) {
                            for (int kw = 0; kw < kernelW; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < inH &&
                                    iw >= 0 && iw < inW) {
                                    sum += input[ic][ih][iw]
                                         * weights[oc][ic][kh][kw];
                                }
                            }
                        }
                    }

                    output[oc][oh][ow] = sum;
                }
            }
        }

        return output;
    }

    /**
     * Backpropagates through the input cached by the last
     * {@link #forward}. Adds this example's weight and bias gradients
     * to the accumulators.
     *
     * @param dOut gradient w.r.t. the output, [outChannels][outH][outW]
     * @return gradient w.r.t. the input, [inChannels][H][W]
     * @throws IllegalStateException if forward has not been called
     */
    public float[][][] backward(float[][][] dOut) {
        if (inputCache == null) {
            throw new IllegalStateException("Call forward before backward");
        }

        int inH = inputCache[0].length;
        int inW = inputCache[0][0].length;
        int outH = dOut[0].length;
        int outW = dOut[0][0].length;

        if (stride == 1) {
            return backwardStride1(dOut, inH, inW, outH, outW);
        }

        float[][][] dIn = new float[inChannels][inH][inW];

        for (int oc = 0; oc < outChannels; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float g = dOut[oc][oh][ow];
                    biasGrads[oc] += g;

                    for (int ic = 0; ic < inChannels; ic++) {
                        for (int kh = 0; kh < kernelH; kh++) {
                            for (int kw = 0; kw < kernelW; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < inH &&
                                    iw >= 0 && iw < inW) {
                                    weightGrads[oc][ic][kh][kw] +=
                                        g * inputCache[ic][ih][iw];
                                    dIn[ic][ih][iw] +=
                                        g * weights[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                }
            }
        }

        return dIn;
    }

    /**
     * Stride-1 forward pass on zero-padded planes. Every padded row has
     * width wp = inW + 2*padding, so output position (oh, ow) lives at
     * flat index oh*wp + ow and kernel tap (kh, kw) reads input at flat
     * index (oh*wp + ow) + kh*wp + kw. Each tap is therefore one
     * contiguous vector operation over the whole plane. Columns
     * ow >= outW are computed but discarded.
     */
    private float[][][] forwardStride1(float[][][] input, int inH,
                                       int inW, int outH, int outW) {
        int wp = inW + 2 * padding;
        int hp = inH + 2 * padding;
        int planeOut = outH * wp;

        paddedCache = new float[inChannels][hp * wp + kernelW];
        for (int ic = 0; ic < inChannels; ic++) {
            for (int ih = 0; ih < inH; ih++) {
                System.arraycopy(input[ic][ih], 0, paddedCache[ic],
                    (ih + padding) * wp + padding, inW);
            }
        }

        float[][][] output = new float[outChannels][outH][outW];
        float[] acc = new float[planeOut];

        for (int oc = 0; oc < outChannels; oc++) {
            Arrays.fill(acc, biases[oc]);
            for (int ic = 0; ic < inChannels; ic++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        VectorMath.axpyRange(acc, 0,
                            weights[oc][ic][kh][kw],
                            paddedCache[ic], kh * wp + kw, planeOut);
                    }
                }
            }
            for (int oh = 0; oh < outH; oh++) {
                System.arraycopy(acc, oh * wp, output[oc][oh], 0, outW);
            }
        }

        return output;
    }

    /**
     * Stride-1 backward pass, the plane-vectorized counterpart of the
     * general loop in {@link #backward}. Gradients for the discarded
     * columns are zero, so they add nothing to the padded input
     * gradient.
     */
    private float[][][] backwardStride1(float[][][] dOut, int inH,
                                        int inW, int outH, int outW) {
        int wp = inW + 2 * padding;
        int hp = inH + 2 * padding;
        int planeOut = outH * wp;

        float[][] dPadded = new float[inChannels][hp * wp + kernelW];
        float[] g = new float[planeOut];

        for (int oc = 0; oc < outChannels; oc++) {
            Arrays.fill(g, 0.0f);
            float rowSum = 0.0f;
            for (int oh = 0; oh < outH; oh++) {
                System.arraycopy(dOut[oc][oh], 0, g, oh * wp, outW);
                for (int ow = 0; ow < outW; ow++) {
                    rowSum += dOut[oc][oh][ow];
                }
            }
            biasGrads[oc] += rowSum;

            for (int ic = 0; ic < inChannels; ic++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        int shift = kh * wp + kw;
                        weightGrads[oc][ic][kh][kw] += VectorMath.dotRange(
                            g, 0, paddedCache[ic], shift, planeOut);
                        VectorMath.axpyRange(dPadded[ic], shift,
                            weights[oc][ic][kh][kw], g, 0, planeOut);
                    }
                }
            }
        }

        float[][][] dIn = new float[inChannels][inH][inW];
        for (int ic = 0; ic < inChannels; ic++) {
            for (int ih = 0; ih < inH; ih++) {
                System.arraycopy(dPadded[ic],
                    (ih + padding) * wp + padding, dIn[ic][ih], 0, inW);
            }
        }

        return dIn;
    }

    /**
     * Applies the accumulated gradients with plain SGD, averaged over
     * the batch.
     *
     * @param learningRate step size
     * @param batchSize    number of examples summed into the gradients
     */
    public void update(float learningRate, int batchSize) {
        float scale = learningRate / batchSize;
        for (int oc = 0; oc < outChannels; oc++) {
            biases[oc] -= scale * biasGrads[oc];
            for (int ic = 0; ic < inChannels; ic++) {
                for (int i = 0; i < kernelH; i++) {
                    for (int j = 0; j < kernelW; j++) {
                        weights[oc][ic][i][j] -=
                            scale * weightGrads[oc][ic][i][j];
                    }
                }
            }
        }
    }

    /** Clears the accumulated gradients. */
    public void zeroGrad() {
        for (int oc = 0; oc < outChannels; oc++) {
            biasGrads[oc] = 0.0f;
            for (int ic = 0; ic < inChannels; ic++) {
                for (int i = 0; i < kernelH; i++) {
                    for (int j = 0; j < kernelW; j++) {
                        weightGrads[oc][ic][i][j] = 0.0f;
                    }
                }
            }
        }
    }

    /** @return number of trainable parameters (weights and biases) */
    public int parameterCount() {
        return outChannels * inChannels * kernelH * kernelW + outChannels;
    }

    /**
     * Copies weights then biases into {@code buf} starting at {@code off}.
     * Layout matches {@link #copyParametersFrom}.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersTo(float[] buf, int off) {
        int k = off;
        for (int oc = 0; oc < outChannels; oc++) {
            for (int ic = 0; ic < inChannels; ic++) {
                for (int i = 0; i < kernelH; i++) {
                    for (int j = 0; j < kernelW; j++) {
                        buf[k++] = weights[oc][ic][i][j];
                    }
                }
            }
        }
        for (int oc = 0; oc < outChannels; oc++) {
            buf[k++] = biases[oc];
        }
    }

    /**
     * Inverse of {@link #copyParametersTo}.
     *
     * @param buf source, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersFrom(float[] buf, int off) {
        int k = off;
        for (int oc = 0; oc < outChannels; oc++) {
            for (int ic = 0; ic < inChannels; ic++) {
                for (int i = 0; i < kernelH; i++) {
                    for (int j = 0; j < kernelW; j++) {
                        weights[oc][ic][i][j] = buf[k++];
                    }
                }
            }
        }
        for (int oc = 0; oc < outChannels; oc++) {
            biases[oc] = buf[k++];
        }
    }

    /**
     * Copies accumulated weight then bias gradients into {@code buf}.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyGradientsTo(float[] buf, int off) {
        int k = off;
        for (int oc = 0; oc < outChannels; oc++) {
            for (int ic = 0; ic < inChannels; ic++) {
                for (int i = 0; i < kernelH; i++) {
                    for (int j = 0; j < kernelW; j++) {
                        buf[k++] = weightGrads[oc][ic][i][j];
                    }
                }
            }
        }
        for (int oc = 0; oc < outChannels; oc++) {
            buf[k++] = biasGrads[oc];
        }
    }

    /**
     * Inverse of {@link #copyGradientsTo}; overwrites the accumulators.
     *
     * @param buf source, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyGradientsFrom(float[] buf, int off) {
        int k = off;
        for (int oc = 0; oc < outChannels; oc++) {
            for (int ic = 0; ic < inChannels; ic++) {
                for (int i = 0; i < kernelH; i++) {
                    for (int j = 0; j < kernelW; j++) {
                        weightGrads[oc][ic][i][j] = buf[k++];
                    }
                }
            }
        }
        for (int oc = 0; oc < outChannels; oc++) {
            biasGrads[oc] = buf[k++];
        }
    }

    /** Number of output channels. */
    public int outChannels() {
        return outChannels;
    }
}
