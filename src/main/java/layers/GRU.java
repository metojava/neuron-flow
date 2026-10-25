package layers;

import java.util.Random;

/**
 * GRU cell: gated recurrent unit with reset and update gates.
 * h_t = (1 - z_t) * h_{t-1} + z_t * n_t
 *
 * <p>Two candidate formulations are supported:
 * <ul>
 *   <li>Default (reset before the recurrent product, Cho et al. 2014):
 *       n_t = tanh(W_n [r_t * h_{t-1}, x_t] + b_n).</li>
 *   <li>Reset after (PyTorch and ONNX default):
 *       n_t = tanh(W_nx x_t + b_nx + r_t * (W_nh h_{t-1} + b_nh)).
 *       Uses one extra recurrent bias b_nh.</li>
 * </ul>
 *
 * <p>W rows are ordered reset, update, candidate. Parameters are laid out as
 * W (row-major), then b, then b_nh (reset-after only).
 */
public final class GRU implements RecurrentCell {

    private final int inputSize;
    private final int hiddenSize;
    private final boolean resetAfter;

    // W: (3*hidden, input+hidden), b: (3*hidden), bhn: (hidden), used only if resetAfter
    private final float[][] W;
    private final float[] b;
    private final float[] bhn;
    private final float[][] Wgrad;
    private final float[] bgrad;
    private final float[] bhnGrad;

    private float[][] inputs;
    private float[][] hStates;
    private float[][] rGate;   // reset gate activations  [seqLen][hidden]
    private float[][] zGate;   // update gate activations [seqLen][hidden]
    private float[][] nGate;   // candidate activations   [seqLen][hidden]
    private float[][] nhCache; // reset-after: W_nh h + b_nh, before the reset gate

    /**
     * Reset-before GRU (the original formulation).
     *
     * @param inputSize  size of each input vector
     * @param hiddenSize size of the hidden state
     * @param seed       RNG seed
     */
    public GRU(int inputSize, int hiddenSize, long seed) {
        this(inputSize, hiddenSize, false, seed);
    }

    /**
     * @param inputSize  size of each input vector
     * @param hiddenSize size of the hidden state
     * @param resetAfter true for the PyTorch/ONNX formulation (reset applied after
     *                   the recurrent matrix product, with a separate recurrent bias)
     * @param seed       RNG seed
     */
    public GRU(int inputSize, int hiddenSize, boolean resetAfter, long seed) {
        if (inputSize <= 0 || hiddenSize <= 0) {
            throw new IllegalArgumentException("sizes must be positive");
        }
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.resetAfter = resetAfter;
        int total = 3 * hiddenSize;
        int inDim = inputSize + hiddenSize;
        this.W = new float[total][inDim];
        this.b = new float[total];
        this.bhn = new float[hiddenSize];
        this.Wgrad = new float[total][inDim];
        this.bgrad = new float[total];
        this.bhnGrad = new float[hiddenSize];

        Random rand = new Random(seed);
        float scale = (float) Math.sqrt(2.0 / inDim);
        for (int i = 0; i < total; i++) {
            for (int j = 0; j < inDim; j++) {
                W[i][j] = (float) (rand.nextGaussian() * scale);
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
        int h = hiddenSize;
        int inDim = inputSize + h;
        inputs = sequence;
        hStates = new float[seqLen + 1][h];
        rGate = new float[seqLen][h];
        zGate = new float[seqLen][h];
        nGate = new float[seqLen][h];
        nhCache = new float[seqLen][h];
        float[][] output = new float[seqLen][h];

        for (int t = 0; t < seqLen; t++) {
            float[] xh = new float[inDim];
            System.arraycopy(sequence[t], 0, xh, 0, inputSize);
            System.arraycopy(hStates[t], 0, xh, inputSize, h);

            float[] r = new float[h];
            float[] z = new float[h];
            float[] n = new float[h];

            for (int i = 0; i < h; i++) {
                float sumR = b[i];
                float sumZ = b[h + i];
                for (int j = 0; j < inDim; j++) {
                    sumR += W[i][j] * xh[j];
                    sumZ += W[h + i][j] * xh[j];
                }
                r[i] = sigmoid(sumR);
                z[i] = sigmoid(sumZ);
            }

            for (int i = 0; i < h; i++) {
                float sumN = b[2 * h + i];
                for (int j = 0; j < inputSize; j++) {
                    sumN += W[2 * h + i][j] * sequence[t][j];
                }
                if (resetAfter) {
                    float hPart = bhn[i];
                    for (int j = 0; j < h; j++) {
                        hPart += W[2 * h + i][inputSize + j] * hStates[t][j];
                    }
                    nhCache[t][i] = hPart;
                    sumN += r[i] * hPart;
                } else {
                    // reset before: element-wise r * h_{t-1}
                    for (int j = 0; j < h; j++) {
                        sumN += W[2 * h + i][inputSize + j] * (r[j] * hStates[t][j]);
                    }
                }
                n[i] = (float) Math.tanh(sumN);

                hStates[t + 1][i] = (1 - z[i]) * hStates[t][i] + z[i] * n[i];
                output[t][i] = hStates[t + 1][i];
            }
            rGate[t] = r;
            zGate[t] = z;
            nGate[t] = n;
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
        float[][] dInput = new float[seqLen][inputSize];
        float[] dhNext = new float[h];
        float[] xh = new float[inDim];

        for (int t = seqLen - 1; t >= 0; t--) {
            float[] x = inputs[t];
            float[] hPrev = hStates[t];
            float[] r = rGate[t];
            float[] z = zGate[t];
            float[] n = nGate[t];
            System.arraycopy(x, 0, xh, 0, inputSize);
            System.arraycopy(hPrev, 0, xh, inputSize, h);

            float[] dh = new float[h];
            float[] daN = new float[h];
            float[] daZ = new float[h];
            float[] daR = new float[h];
            for (int i = 0; i < h; i++) {
                dh[i] = dOutput[t][i] + dhNext[i];
                daN[i] = dh[i] * z[i] * (1.0f - n[i] * n[i]);
                daZ[i] = dh[i] * (n[i] - hPrev[i]) * z[i] * (1.0f - z[i]);
            }

            // Gradient reaching h_{t-1} through the candidate's recurrent product
            float[] dhFromCand = new float[h];
            if (resetAfter) {
                for (int i = 0; i < h; i++) {
                    float dPart = daN[i] * r[i];            // d(W_nh h + b_nh)
                    daR[i] = daN[i] * nhCache[t][i] * r[i] * (1.0f - r[i]);
                    bhnGrad[i] += dPart;
                    for (int k = 0; k < h; k++) {
                        Wgrad[2 * h + i][inputSize + k] += dPart * hPrev[k];
                        dhFromCand[k] += W[2 * h + i][inputSize + k] * dPart;
                    }
                }
            } else {
                float[] dRH = new float[h];
                for (int k = 0; k < h; k++) {
                    float sum = 0.0f;
                    for (int i = 0; i < h; i++) {
                        sum += W[2 * h + i][inputSize + k] * daN[i];
                    }
                    dRH[k] = sum;
                }
                for (int k = 0; k < h; k++) {
                    daR[k] = dRH[k] * hPrev[k] * r[k] * (1.0f - r[k]);
                    dhFromCand[k] = dRH[k] * r[k];
                }
                for (int i = 0; i < h; i++) {
                    for (int k = 0; k < h; k++) {
                        Wgrad[2 * h + i][inputSize + k] += daN[i] * (r[k] * hPrev[k]);
                    }
                }
            }

            // Weight and bias gradients for the reset, update and candidate rows
            for (int i = 0; i < h; i++) {
                bgrad[i] += daR[i];
                bgrad[h + i] += daZ[i];
                bgrad[2 * h + i] += daN[i];
                for (int j = 0; j < inDim; j++) {
                    Wgrad[i][j] += daR[i] * xh[j];
                    Wgrad[h + i][j] += daZ[i] * xh[j];
                }
                for (int j = 0; j < inputSize; j++) {
                    Wgrad[2 * h + i][j] += daN[i] * x[j];
                }
            }

            // Gradients w.r.t. [x_t, h_{t-1}] through the reset and update gates
            float[] dxh = new float[inDim];
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < inDim; j++) {
                    dxh[j] += W[i][j] * daR[i] + W[h + i][j] * daZ[i];
                }
            }
            // Gradient w.r.t. x_t through the candidate
            float[] dx = new float[inputSize];
            for (int j = 0; j < inputSize; j++) {
                float sum = dxh[j];
                for (int i = 0; i < h; i++) {
                    sum += W[2 * h + i][j] * daN[i];
                }
                dx[j] = sum;
            }
            System.arraycopy(dx, 0, dInput[t], 0, inputSize);

            // Gradient w.r.t. h_{t-1}: direct leak, gates, and candidate
            for (int k = 0; k < h; k++) {
                dhNext[k] = dh[k] * (1.0f - z[k]) + dhFromCand[k] + dxh[inputSize + k];
            }
        }
        return dInput;
    }

    private int totalRows() {
        return 3 * hiddenSize;
    }

    /**
     * Copies weights (row-major, one row per gate unit), then biases, then
     * the reset-after recurrent bias if present.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersTo(float[] buf, int off) {
        int inDim = inputSize + hiddenSize;
        int total = totalRows();
        for (int r = 0; r < total; r++) {
            System.arraycopy(W[r], 0, buf, off + r * inDim, inDim);
        }
        int p = off + total * inDim;
        System.arraycopy(b, 0, buf, p, total);
        if (resetAfter) {
            System.arraycopy(bhn, 0, buf, p + total, hiddenSize);
        }
    }

    /**
     * Inverse of {@link #copyParametersTo}.
     *
     * @param buf source, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersFrom(float[] buf, int off) {
        int inDim = inputSize + hiddenSize;
        int total = totalRows();
        for (int r = 0; r < total; r++) {
            System.arraycopy(buf, off + r * inDim, W[r], 0, inDim);
        }
        int p = off + total * inDim;
        System.arraycopy(buf, p, b, 0, total);
        if (resetAfter) {
            System.arraycopy(buf, p + total, bhn, 0, hiddenSize);
        }
    }

    /**
     * Copies accumulated gradients in the same layout as {@link #copyParametersTo}.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyGradientsTo(float[] buf, int off) {
        int inDim = inputSize + hiddenSize;
        int total = totalRows();
        for (int r = 0; r < total; r++) {
            System.arraycopy(Wgrad[r], 0, buf, off + r * inDim, inDim);
        }
        int p = off + total * inDim;
        System.arraycopy(bgrad, 0, buf, p, total);
        if (resetAfter) {
            System.arraycopy(bhnGrad, 0, buf, p + total, hiddenSize);
        }
    }

    /** Applies accumulated gradients with SGD. */
    public void update(float learningRate, int batchSize) {
        float scale = learningRate / batchSize;
        int total = totalRows();
        int inDim = inputSize + hiddenSize;
        for (int i = 0; i < total; i++) {
            b[i] -= scale * bgrad[i];
            for (int j = 0; j < inDim; j++) {
                W[i][j] -= scale * Wgrad[i][j];
            }
        }
        if (resetAfter) {
            for (int i = 0; i < hiddenSize; i++) {
                bhn[i] -= scale * bhnGrad[i];
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

    /** @return true if this cell uses the reset-after formulation */
    public boolean isResetAfter() {
        return resetAfter;
    }

    /** Clears accumulated gradients. */
    public void zeroGrad() {
        int total = totalRows();
        int inDim = inputSize + hiddenSize;
        for (int i = 0; i < total; i++) {
            bgrad[i] = 0.0f;
            for (int j = 0; j < inDim; j++) {
                Wgrad[i][j] = 0.0f;
            }
        }
        java.util.Arrays.fill(bhnGrad, 0.0f);
    }

    public int parameterCount() {
        int inDim = inputSize + hiddenSize;
        int count = 3 * hiddenSize * inDim + 3 * hiddenSize;
        return resetAfter ? count + hiddenSize : count;
    }
}
