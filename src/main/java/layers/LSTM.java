package layers;

import java.util.Random;

/**
 * LSTM cell: long short-term memory with forget, input, and output gates.
 * h_t = o_t * tanh(c_t), c_t = f_t * c_{t-1} + i_t * g_t
 */
public final class LSTM implements RecurrentCell {

    private final int inputSize;
    private final int hiddenSize;

    // Weights: W (4*hidden, input+hidden), b (4*hidden)
    private final float[][] W;  // [4*hidden][input+hidden]
    private final float[] b;
    private final float[][] Wgrad;
    private final float[] bgrad;

    private float[][] inputs;
    private float[][] hStates;
    private float[][] cStates;
    private float[][] gates; // [seqLen][4*hidden]

    /**
     * @param inputSize  size of each input vector
     * @param hiddenSize size of the hidden state
     * @param seed       RNG seed
     */
    public LSTM(int inputSize, int hiddenSize, long seed) {
        if (inputSize <= 0 || hiddenSize <= 0) {
            throw new IllegalArgumentException("sizes must be positive");
        }
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        int total = 4 * hiddenSize;
        int inDim = inputSize + hiddenSize;
        this.W = new float[total][inDim];
        this.b = new float[total];
        this.Wgrad = new float[total][inDim];
        this.bgrad = new float[total];

        Random rand = new Random(seed);
        float scale = (float) Math.sqrt(2.0 / inDim);
        for (int i = 0; i < total; i++) {
            for (int j = 0; j < inDim; j++) {
                W[i][j] = (float) (rand.nextGaussian() * scale);
            }
            // Forget gate bias initialized to 1
            if (i < hiddenSize) {
                b[i] = 1.0f;
            }
        }
    }

    private static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    /**
     * Forward pass over a sequence.
     *
     * @param sequence [seqLen][inputSize]
     * @return [seqLen][hiddenSize] hidden states
     */
    public float[][] forward(float[][] sequence) {
        int seqLen = sequence.length;
        int inDim = inputSize + hiddenSize;
        inputs = sequence;
        hStates = new float[seqLen + 1][hiddenSize];
        cStates = new float[seqLen + 1][hiddenSize];
        gates = new float[seqLen][4 * hiddenSize];
        float[][] output = new float[seqLen][hiddenSize];

        for (int t = 0; t < seqLen; t++) {
            // Concatenate input and previous hidden state
            float[] xh = new float[inDim];
            System.arraycopy(sequence[t], 0, xh, 0, inputSize);
            System.arraycopy(hStates[t], 0, xh, inputSize, hiddenSize);

            // Compute gates: i, f, g, o
            for (int i = 0; i < 4 * hiddenSize; i++) {
                float sum = b[i];
                for (int j = 0; j < inDim; j++) {
                    sum += W[i][j] * xh[j];
                }
                gates[t][i] = sum;
            }

            // Apply activations
            for (int i = 0; i < hiddenSize; i++) {
                float ft = sigmoid(gates[t][i]);                    // forget
                float it = sigmoid(gates[t][hiddenSize + i]);       // input
                float gt = (float) Math.tanh(gates[t][2 * hiddenSize + i]); // cell candidate
                float ot = sigmoid(gates[t][3 * hiddenSize + i]);    // output

                cStates[t + 1][i] = ft * cStates[t][i] + it * gt;
                hStates[t + 1][i] = ot * (float) Math.tanh(cStates[t + 1][i]);
                output[t][i] = hStates[t + 1][i];
            }
        }
        return output;
    }

    /**
     * Backpropagation through time for the last {@link #forward} call.
     * Accumulates weight and bias gradients (applied by {@link #update}).
     *
     * @param dOutput [seqLen][hiddenSize] gradient w.r.t. the outputs
     * @return [seqLen][inputSize] gradient w.r.t. the inputs
     * @throws IllegalStateException    if forward has not been called
     * @throws IllegalArgumentException if dOutput does not match the sequence length
     */
    public float[][] backward(float[][] dOutput) {
        if (inputs == null) {
            throw new IllegalStateException("Call forward before backward");
        }
        int seqLen = inputs.length;
        if (dOutput.length != seqLen) {
            throw new IllegalArgumentException(
                "Expected " + seqLen + " output gradients, got " + dOutput.length);
        }
        int h = hiddenSize;
        int inDim = inputSize + h;
        int total = 4 * h;
        float[][] dInput = new float[seqLen][inputSize];
        float[] dhNext = new float[h];
        float[] dcNext = new float[h];
        float[] xh = new float[inDim];
        float[] dpre = new float[total];

        for (int t = seqLen - 1; t >= 0; t--) {
            System.arraycopy(inputs[t], 0, xh, 0, inputSize);
            System.arraycopy(hStates[t], 0, xh, inputSize, h);

            for (int i = 0; i < h; i++) {
                float f = sigmoid(gates[t][i]);
                float in = sigmoid(gates[t][h + i]);
                float g = (float) Math.tanh(gates[t][2 * h + i]);
                float o = sigmoid(gates[t][3 * h + i]);
                float tc = (float) Math.tanh(cStates[t + 1][i]);

                float dh = dOutput[t][i] + dhNext[i];
                float dc = dcNext[i] + dh * o * (1.0f - tc * tc);

                dpre[i] = dc * cStates[t][i] * f * (1.0f - f);      // forget gate
                dpre[h + i] = dc * g * in * (1.0f - in);            // input gate
                dpre[2 * h + i] = dc * in * (1.0f - g * g);         // cell candidate
                dpre[3 * h + i] = dh * tc * o * (1.0f - o);         // output gate
                dcNext[i] = dc * f;
            }

            for (int r = 0; r < total; r++) {
                bgrad[r] += dpre[r];
                for (int j = 0; j < inDim; j++) {
                    Wgrad[r][j] += dpre[r] * xh[j];
                }
            }

            float[] dxh = new float[inDim];
            for (int r = 0; r < total; r++) {
                for (int j = 0; j < inDim; j++) {
                    dxh[j] += W[r][j] * dpre[r];
                }
            }
            System.arraycopy(dxh, 0, dInput[t], 0, inputSize);
            System.arraycopy(dxh, inputSize, dhNext, 0, h);
        }
        return dInput;
    }

    /**
     * Copies weights (row-major, one row per gate unit) then biases into buf.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersTo(float[] buf, int off) {
        int inDim = inputSize + hiddenSize;
        int total = 4 * hiddenSize;
        for (int r = 0; r < total; r++) {
            System.arraycopy(W[r], 0, buf, off + r * inDim, inDim);
        }
        System.arraycopy(b, 0, buf, off + total * inDim, total);
    }

    /**
     * Inverse of {@link #copyParametersTo}.
     *
     * @param buf source, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersFrom(float[] buf, int off) {
        int inDim = inputSize + hiddenSize;
        int total = 4 * hiddenSize;
        for (int r = 0; r < total; r++) {
            System.arraycopy(buf, off + r * inDim, W[r], 0, inDim);
        }
        System.arraycopy(buf, off + total * inDim, b, 0, total);
    }

    /**
     * Copies accumulated gradients in the same layout as {@link #copyParametersTo}.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyGradientsTo(float[] buf, int off) {
        int inDim = inputSize + hiddenSize;
        int total = 4 * hiddenSize;
        for (int r = 0; r < total; r++) {
            System.arraycopy(Wgrad[r], 0, buf, off + r * inDim, inDim);
        }
        System.arraycopy(bgrad, 0, buf, off + total * inDim, total);
    }

    /** Applies accumulated gradients with SGD. */
    public void update(float learningRate, int batchSize) {
        float scale = learningRate / batchSize;
        int total = 4 * hiddenSize;
        int inDim = inputSize + hiddenSize;
        for (int i = 0; i < total; i++) {
            b[i] -= scale * bgrad[i];
            for (int j = 0; j < inDim; j++) {
                W[i][j] -= scale * Wgrad[i][j];
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
        int total = 4 * hiddenSize;
        int inDim = inputSize + hiddenSize;
        for (int i = 0; i < total; i++) {
            bgrad[i] = 0.0f;
            for (int j = 0; j < inDim; j++) {
                Wgrad[i][j] = 0.0f;
            }
        }
    }

    public int parameterCount() {
        int inDim = inputSize + hiddenSize;
        return 4 * hiddenSize * inDim + 4 * hiddenSize;
    }
}
