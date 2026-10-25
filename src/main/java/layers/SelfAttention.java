package layers;

import java.util.Random;

/**
 * Scaled dot-product self-attention (single head). Given an input
 * sequence of vectors, computes attention-weighted outputs.
 *
 * <p>Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V, followed by an output projection.
 * Projection matrices are [dim][dim] with rows indexed by output, the same layout as
 * PyTorch's nn.Linear weights. The layer has no bias terms.
 */
public final class SelfAttention {

    private final int dim;
    private final float[][] Wq;  // [dim][dim]
    private final float[][] Wk;
    private final float[][] Wv;
    private final float[][] Wo;  // output projection

    private final float[][] Wqg;
    private final float[][] Wkg;
    private final float[][] Wvg;
    private final float[][] Wog;

    // Forward-pass caches for backward
    private float[][] lastInput;
    private float[][] Q;
    private float[][] K;
    private float[][] V;
    private float[][] weights;   // softmax attention weights [seqLen][seqLen]
    private float[][] attended;  // weighted sums of V [seqLen][dim]

    /**
     * @param dim  model dimension (input and output size per token)
     * @param seed RNG seed
     */
    public SelfAttention(int dim, long seed) {
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be positive");
        }
        this.dim = dim;
        this.Wq = new float[dim][dim];
        this.Wk = new float[dim][dim];
        this.Wv = new float[dim][dim];
        this.Wo = new float[dim][dim];
        this.Wqg = new float[dim][dim];
        this.Wkg = new float[dim][dim];
        this.Wvg = new float[dim][dim];
        this.Wog = new float[dim][dim];

        Random rand = new Random(seed);
        float scale = (float) Math.sqrt(1.0 / dim);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                Wq[i][j] = (float) (rand.nextGaussian() * scale);
                Wk[i][j] = (float) (rand.nextGaussian() * scale);
                Wv[i][j] = (float) (rand.nextGaussian() * scale);
                Wo[i][j] = (float) (rand.nextGaussian() * scale);
            }
        }
    }

    /**
     * Forward pass.
     *
     * @param input [seqLen][dim] input sequence
     * @return [seqLen][dim] attention output
     */
    public float[][] forward(float[][] input) {
        int seqLen = input.length;
        lastInput = input;
        float scale = (float) (1.0 / Math.sqrt(dim));

        Q = new float[seqLen][dim];
        K = new float[seqLen][dim];
        V = new float[seqLen][dim];
        for (int t = 0; t < seqLen; t++) {
            for (int d = 0; d < dim; d++) {
                float q = 0, k = 0, v = 0;
                for (int j = 0; j < dim; j++) {
                    q += Wq[d][j] * input[t][j];
                    k += Wk[d][j] * input[t][j];
                    v += Wv[d][j] * input[t][j];
                }
                Q[t][d] = q;
                K[t][d] = k;
                V[t][d] = v;
            }
        }

        float[][] scores = new float[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                float dot = 0;
                for (int d = 0; d < dim; d++) {
                    dot += Q[i][d] * K[j][d];
                }
                scores[i][j] = dot * scale;
            }
        }

        weights = new float[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            float max = scores[i][0];
            for (int j = 1; j < seqLen; j++) {
                if (scores[i][j] > max) max = scores[i][j];
            }
            float sum = 0;
            for (int j = 0; j < seqLen; j++) {
                weights[i][j] = (float) Math.exp(scores[i][j] - max);
                sum += weights[i][j];
            }
            for (int j = 0; j < seqLen; j++) {
                weights[i][j] /= sum;
            }
        }

        attended = new float[seqLen][dim];
        for (int i = 0; i < seqLen; i++) {
            for (int d = 0; d < dim; d++) {
                float sum = 0;
                for (int j = 0; j < seqLen; j++) {
                    sum += weights[i][j] * V[j][d];
                }
                attended[i][d] = sum;
            }
        }

        float[][] out = new float[seqLen][dim];
        for (int t = 0; t < seqLen; t++) {
            for (int d = 0; d < dim; d++) {
                float sum = 0;
                for (int j = 0; j < dim; j++) {
                    sum += Wo[d][j] * attended[t][j];
                }
                out[t][d] = sum;
            }
        }
        return out;
    }

    /**
     * Backpropagation through the last {@link #forward} call. Accumulates
     * gradients for all four projections.
     *
     * @param dOut [seqLen][dim] gradient w.r.t. the output
     * @return [seqLen][dim] gradient w.r.t. the input
     * @throws IllegalStateException    if forward has not been called
     * @throws IllegalArgumentException if dOut does not match the sequence length
     */
    public float[][] backward(float[][] dOut) {
        if (lastInput == null) {
            throw new IllegalStateException("Call forward before backward");
        }
        int n = lastInput.length;
        if (dOut.length != n) {
            throw new IllegalArgumentException(
                "Expected " + n + " output gradients, got " + dOut.length);
        }
        float scale = (float) (1.0 / Math.sqrt(dim));

        // out = Wo * attended
        float[][] dAtt = new float[n][dim];
        for (int t = 0; t < n; t++) {
            for (int d = 0; d < dim; d++) {
                for (int j = 0; j < dim; j++) {
                    Wog[d][j] += dOut[t][d] * attended[t][j];
                    dAtt[t][j] += Wo[d][j] * dOut[t][d];
                }
            }
        }

        // attended = weights * V
        float[][] dV = new float[n][dim];
        float[][] dW = new float[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float s = 0;
                for (int d = 0; d < dim; d++) {
                    s += dAtt[i][d] * V[j][d];
                    dV[j][d] += weights[i][j] * dAtt[i][d];
                }
                dW[i][j] = s;
            }
        }

        // softmax backward, row by row
        float[][] dScore = new float[n][n];
        for (int i = 0; i < n; i++) {
            float dot = 0;
            for (int k = 0; k < n; k++) dot += weights[i][k] * dW[i][k];
            for (int j = 0; j < n; j++) {
                dScore[i][j] = weights[i][j] * (dW[i][j] - dot);
            }
        }

        // scores = scale * Q K^T
        float[][] dQ = new float[n][dim];
        float[][] dK = new float[n][dim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float g = scale * dScore[i][j];
                for (int d = 0; d < dim; d++) {
                    dQ[i][d] += g * K[j][d];
                    dK[j][d] += g * Q[i][d];
                }
            }
        }

        // Q, K, V = projections of the input
        float[][] dX = new float[n][dim];
        for (int t = 0; t < n; t++) {
            for (int d = 0; d < dim; d++) {
                for (int j = 0; j < dim; j++) {
                    float x = lastInput[t][j];
                    Wqg[d][j] += dQ[t][d] * x;
                    Wkg[d][j] += dK[t][d] * x;
                    Wvg[d][j] += dV[t][d] * x;
                    dX[t][j] += Wq[d][j] * dQ[t][d]
                              + Wk[d][j] * dK[t][d]
                              + Wv[d][j] * dV[t][d];
                }
            }
        }
        return dX;
    }

    /** @return model dimension */
    public int dim() {
        return dim;
    }

    /**
     * Replaces the four projection matrices, for example with weights loaded
     * from a model file. Each matrix is [dim][dim] with rows indexed by output.
     *
     * @throws IllegalArgumentException if any matrix has the wrong shape
     */
    public void loadWeights(float[][] q, float[][] k, float[][] v, float[][] o) {
        checkSquare(q, "Wq");
        checkSquare(k, "Wk");
        checkSquare(v, "Wv");
        checkSquare(o, "Wo");
        copy(q, Wq);
        copy(k, Wk);
        copy(v, Wv);
        copy(o, Wo);
    }

    private void checkSquare(float[][] m, String name) {
        if (m.length != dim) {
            throw new IllegalArgumentException(name + ": expected " + dim + " rows, got " + m.length);
        }
        for (float[] row : m) {
            if (row.length != dim) {
                throw new IllegalArgumentException(name + ": expected rows of " + dim + " values");
            }
        }
    }

    private static void copy(float[][] src, float[][] dst) {
        for (int i = 0; i < src.length; i++) {
            System.arraycopy(src[i], 0, dst[i], 0, src[i].length);
        }
    }

    /** @return number of trainable parameters (four projections) */
    public int parameterCount() {
        return 4 * dim * dim;
    }

    /**
     * Copies Wq, Wk, Wv, Wo (each row-major) into buf.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersTo(float[] buf, int off) {
        copyMatrices(buf, off, Wq, Wk, Wv, Wo, false);
    }

    /**
     * Inverse of {@link #copyParametersTo}.
     *
     * @param buf source, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersFrom(float[] buf, int off) {
        copyMatrices(buf, off, Wq, Wk, Wv, Wo, true);
    }

    /**
     * Copies accumulated gradients in the same layout as {@link #copyParametersTo}.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyGradientsTo(float[] buf, int off) {
        copyMatrices(buf, off, Wqg, Wkg, Wvg, Wog, false);
    }

    private void copyMatrices(float[] buf, int off, float[][] a, float[][] b,
                              float[][] c, float[][] d, boolean from) {
        int p = off;
        for (float[][] m : new float[][][] {a, b, c, d}) {
            for (int i = 0; i < dim; i++) {
                if (from) {
                    System.arraycopy(buf, p, m[i], 0, dim);
                } else {
                    System.arraycopy(m[i], 0, buf, p, dim);
                }
                p += dim;
            }
        }
    }

    /** Applies accumulated gradients with SGD. */
    public void update(float learningRate, int batchSize) {
        float scale = -learningRate / batchSize;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                Wq[i][j] += scale * Wqg[i][j];
                Wk[i][j] += scale * Wkg[i][j];
                Wv[i][j] += scale * Wvg[i][j];
                Wo[i][j] += scale * Wog[i][j];
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

    /** Clears accumulated gradients. */
    public void zeroGrad() {
        for (float[][] m : new float[][][] {Wqg, Wkg, Wvg, Wog}) {
            for (float[] row : m) java.util.Arrays.fill(row, 0.0f);
        }
    }
}
