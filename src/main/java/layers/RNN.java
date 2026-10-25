package layers;

import java.util.Random;

/**
 * Vanilla RNN cell: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b).
 * Processes a sequence of inputs and returns the sequence of hidden states.
 */
public final class RNN implements RecurrentCell {

    private final int inputSize;
    private final int hiddenSize;
    private final float[][] wxh;  // [hiddenSize][inputSize]
    private final float[][] whh;  // [hiddenSize][hiddenSize]
    private final float[] bh;     // [hiddenSize]

    private float[][] wxhGrad;
    private float[][] whhGrad;
    private float[] bhGrad;

    private float[][] inputs;
    private float[][] hiddenStates;

    /**
     * @param inputSize  size of each input vector
     * @param hiddenSize size of the hidden state
     * @param seed       RNG seed
     */
    public RNN(int inputSize, int hiddenSize, long seed) {
        if (inputSize <= 0 || hiddenSize <= 0) {
            throw new IllegalArgumentException("sizes must be positive");
        }
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.wxh = new float[hiddenSize][inputSize];
        this.whh = new float[hiddenSize][hiddenSize];
        this.bh = new float[hiddenSize];
        this.wxhGrad = new float[hiddenSize][inputSize];
        this.whhGrad = new float[hiddenSize][hiddenSize];
        this.bhGrad = new float[hiddenSize];

        Random rand = new Random(seed);
        float scale = (float) Math.sqrt(2.0 / (inputSize + hiddenSize));
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                wxh[i][j] = (float) (rand.nextGaussian() * scale);
            }
            for (int j = 0; j < hiddenSize; j++) {
                whh[i][j] = (float) (rand.nextGaussian() * scale);
            }
        }
    }

    /**
     * Forward pass over a sequence.
     *
     * @param sequence [seqLen][inputSize]
     * @return [seqLen][hiddenSize] hidden states
     */
    public float[][] forward(float[][] sequence) {
        int seqLen = sequence.length;
        inputs = sequence;
        hiddenStates = new float[seqLen + 1][hiddenSize]; // +1 for h_{-1}=0
        float[][] output = new float[seqLen][hiddenSize];

        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < hiddenSize; i++) {
                float sum = bh[i];
                for (int j = 0; j < inputSize; j++) {
                    sum += wxh[i][j] * sequence[t][j];
                }
                for (int j = 0; j < hiddenSize; j++) {
                    sum += whh[i][j] * hiddenStates[t][j];
                }
                hiddenStates[t + 1][i] = (float) Math.tanh(sum);
                output[t][i] = hiddenStates[t + 1][i];
            }
        }
        return output;
    }

    /**
     * Backward pass through time (BPTT). Accumulates gradients.
     *
     * @param dOutput [seqLen][hiddenSize] gradient w.r.t. outputs
     * @return [seqLen][inputSize] gradient w.r.t. inputs
     */
    public float[][] backward(float[][] dOutput) {
        int seqLen = dOutput.length;
        float[][] dInput = new float[seqLen][inputSize];
        float[] dhNext = new float[hiddenSize];

        for (int t = seqLen - 1; t >= 0; t--) {
            float[] dh = new float[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                // Gradient w.r.t. the pre-activation: total gradient on h_t times tanh'(.)
                dh[i] = (dOutput[t][i] + dhNext[i])
                    * (1.0f - hiddenStates[t + 1][i] * hiddenStates[t + 1][i]);
            }

            // Gradients for weights
            for (int i = 0; i < hiddenSize; i++) {
                bhGrad[i] += dh[i];
                for (int j = 0; j < inputSize; j++) {
                    wxhGrad[i][j] += dh[i] * inputs[t][j];
                }
                for (int j = 0; j < hiddenSize; j++) {
                    whhGrad[i][j] += dh[i] * hiddenStates[t][j];
                }
            }

            // Gradient w.r.t. input
            for (int j = 0; j < inputSize; j++) {
                float sum = 0.0f;
                for (int i = 0; i < hiddenSize; i++) {
                    sum += wxh[i][j] * dh[i];
                }
                dInput[t][j] = sum;
            }

            // Gradient w.r.t. previous hidden state
            for (int j = 0; j < hiddenSize; j++) {
                float sum = 0.0f;
                for (int i = 0; i < hiddenSize; i++) {
                    sum += whh[i][j] * dh[i];
                }
                dhNext[j] = sum;
            }
        }
        return dInput;
    }

    /** Applies accumulated gradients with SGD. */
    public void update(float learningRate, int batchSize) {
        float scale = learningRate / batchSize;
        for (int i = 0; i < hiddenSize; i++) {
            bh[i] -= scale * bhGrad[i];
            for (int j = 0; j < inputSize; j++) {
                wxh[i][j] -= scale * wxhGrad[i][j];
            }
            for (int j = 0; j < hiddenSize; j++) {
                whh[i][j] -= scale * whhGrad[i][j];
            }
        }
    }

    /**
     * Applies one step of an optimizer (for example {@link optim.Adam}) to all
     * parameters, using gradients averaged over the batch.
     *
     * @param optimizer update rule
     * @param batchSize number of sequences summed into the gradients
     */
    public void update(optim.Optimizer optimizer, int batchSize) {
        float[] p = new float[parameterCount()];
        float[] g = new float[p.length];
        copyParametersTo(p, 0);
        copyGradientsTo(g, 0);
        for (int i = 0; i < g.length; i++) g[i] /= batchSize;
        optimizer.update(p, g);
        copyParametersFrom(p, 0);
    }

    /**
     * Copies parameters into buf: W_xh rows, then W_hh rows, then bias.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersTo(float[] buf, int off) {
        int p = off;
        for (int i = 0; i < hiddenSize; i++) {
            System.arraycopy(wxh[i], 0, buf, p, inputSize);
            p += inputSize;
        }
        for (int i = 0; i < hiddenSize; i++) {
            System.arraycopy(whh[i], 0, buf, p, hiddenSize);
            p += hiddenSize;
        }
        System.arraycopy(bh, 0, buf, p, hiddenSize);
    }

    /**
     * Inverse of {@link #copyParametersTo}.
     *
     * @param buf source, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersFrom(float[] buf, int off) {
        int p = off;
        for (int i = 0; i < hiddenSize; i++) {
            System.arraycopy(buf, p, wxh[i], 0, inputSize);
            p += inputSize;
        }
        for (int i = 0; i < hiddenSize; i++) {
            System.arraycopy(buf, p, whh[i], 0, hiddenSize);
            p += hiddenSize;
        }
        System.arraycopy(buf, p, bh, 0, hiddenSize);
    }

    /**
     * Copies accumulated gradients in the same layout as {@link #copyParametersTo}.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyGradientsTo(float[] buf, int off) {
        int p = off;
        for (int i = 0; i < hiddenSize; i++) {
            System.arraycopy(wxhGrad[i], 0, buf, p, inputSize);
            p += inputSize;
        }
        for (int i = 0; i < hiddenSize; i++) {
            System.arraycopy(whhGrad[i], 0, buf, p, hiddenSize);
            p += hiddenSize;
        }
        System.arraycopy(bhGrad, 0, buf, p, hiddenSize);
    }

    /** @return input vector size */
    public int inputSize() {
        return inputSize;
    }

    /** @return hidden state size */
    public int hiddenSize() {
        return hiddenSize;
    }

    /** Clears accumulated gradients. */
    public void zeroGrad() {
        for (int i = 0; i < hiddenSize; i++) {
            bhGrad[i] = 0.0f;
            for (int j = 0; j < inputSize; j++) {
                wxhGrad[i][j] = 0.0f;
            }
            for (int j = 0; j < hiddenSize; j++) {
                whhGrad[i][j] = 0.0f;
            }
        }
    }

    public int parameterCount() {
        return hiddenSize * inputSize + hiddenSize * hiddenSize + hiddenSize;
    }
}
