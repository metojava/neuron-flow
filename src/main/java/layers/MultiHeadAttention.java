package layers;

import java.util.Arrays;
import java.util.Random;

/**
 * Multi-head scaled dot-product self-attention.
 *
 * <p>Q, K and V are linear projections of the input (with biases). Each of the
 * {@code heads} heads attends over its own slice of size {@code dim / heads}, and
 * the head outputs are concatenated and passed through an output projection.
 * Projection matrices are [dim][dim] with rows indexed by output, matching PyTorch's
 * {@code nn.MultiheadAttention} (in_proj_weight and out_proj.weight).
 */
public final class MultiHeadAttention {

    private final int dim;
    private final int heads;
    private final int headDim;

    // Parameters: weights [dim][dim] (rows = output), biases [dim]
    private final float[][] Wq, Wk, Wv, Wo;
    private final float[] bq, bk, bv, bo;

    // Gradients
    private final float[][] Wqg, Wkg, Wvg, Wog;
    private final float[] bqg, bkg, bvg, bog;

    // Forward caches
    private float[][] lastInput;
    private float[][] Q, K, V;
    private float[][][] attn;  // [heads][seq][seq]
    private float[][] ctx;     // concatenated head outputs [seq][dim]

    /**
     * @param dim   model dimension
     * @param heads number of heads; must divide dim
     * @param seed  RNG seed
     * @throws IllegalArgumentException if sizes are not positive or heads does not divide dim
     */
    public MultiHeadAttention(int dim, int heads, long seed) {
        if (dim <= 0 || heads <= 0 || dim % heads != 0) {
            throw new IllegalArgumentException("dim and heads must be positive and heads must divide dim");
        }
        this.dim = dim;
        this.heads = heads;
        this.headDim = dim / heads;
        Wq = new float[dim][dim];
        Wk = new float[dim][dim];
        Wv = new float[dim][dim];
        Wo = new float[dim][dim];
        bq = new float[dim];
        bk = new float[dim];
        bv = new float[dim];
        bo = new float[dim];
        Wqg = new float[dim][dim];
        Wkg = new float[dim][dim];
        Wvg = new float[dim][dim];
        Wog = new float[dim][dim];
        bqg = new float[dim];
        bkg = new float[dim];
        bvg = new float[dim];
        bog = new float[dim];

        Random rand = new Random(seed);
        float scale = (float) Math.sqrt(1.0 / dim);
        for (float[][] m : new float[][][] {Wq, Wk, Wv, Wo}) {
            for (float[] row : m) {
                for (int j = 0; j < dim; j++) row[j] = (float) (rand.nextGaussian() * scale);
            }
        }
    }

    /**
     * @param input [seqLen][dim]
     * @return [seqLen][dim] attention output
     */
    public float[][] forward(float[][] input) {
        int n = input.length;
        lastInput = input;
        Q = project(input, Wq, bq);
        K = project(input, Wk, bk);
        V = project(input, Wv, bv);
        float scale = (float) (1.0 / Math.sqrt(headDim));

        attn = new float[heads][n][n];
        ctx = new float[n][dim];
        for (int h = 0; h < heads; h++) {
            int off = h * headDim;
            for (int i = 0; i < n; i++) {
                float max = Float.NEGATIVE_INFINITY;
                float[] score = new float[n];
                for (int j = 0; j < n; j++) {
                    float s = 0;
                    for (int c = 0; c < headDim; c++) s += Q[i][off + c] * K[j][off + c];
                    score[j] = s * scale;
                    max = Math.max(max, score[j]);
                }
                float sum = 0;
                for (int j = 0; j < n; j++) {
                    score[j] = (float) Math.exp(score[j] - max);
                    sum += score[j];
                }
                for (int j = 0; j < n; j++) {
                    attn[h][i][j] = score[j] / sum;
                }
                for (int c = 0; c < headDim; c++) {
                    float s = 0;
                    for (int j = 0; j < n; j++) s += attn[h][i][j] * V[j][off + c];
                    ctx[i][off + c] = s;
                }
            }
        }

        float[][] out = new float[n][dim];
        for (int t = 0; t < n; t++) {
            for (int d = 0; d < dim; d++) {
                float s = bo[d];
                for (int j = 0; j < dim; j++) s += Wo[d][j] * ctx[t][j];
                out[t][d] = s;
            }
        }
        return out;
    }

    private float[][] project(float[][] x, float[][] w, float[] b) {
        int n = x.length;
        float[][] out = new float[n][dim];
        for (int t = 0; t < n; t++) {
            for (int d = 0; d < dim; d++) {
                float s = b[d];
                for (int j = 0; j < dim; j++) s += w[d][j] * x[t][j];
                out[t][d] = s;
            }
        }
        return out;
    }

    /**
     * Backpropagation through the last {@link #forward} call. Accumulates gradients
     * for all weights and biases.
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
            throw new IllegalArgumentException("Expected " + n + " output gradients, got " + dOut.length);
        }
        float scale = (float) (1.0 / Math.sqrt(headDim));

        // out = Wo ctx + bo
        float[][] dCtx = new float[n][dim];
        for (int t = 0; t < n; t++) {
            for (int d = 0; d < dim; d++) {
                bog[d] += dOut[t][d];
                for (int j = 0; j < dim; j++) {
                    Wog[d][j] += dOut[t][d] * ctx[t][j];
                    dCtx[t][j] += Wo[d][j] * dOut[t][d];
                }
            }
        }

        float[][] dQ = new float[n][dim];
        float[][] dK = new float[n][dim];
        float[][] dV = new float[n][dim];
        for (int h = 0; h < heads; h++) {
            int off = h * headDim;
            float[][] A = attn[h];
            // ctx = A V
            float[][] dA = new float[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    float s = 0;
                    for (int c = 0; c < headDim; c++) {
                        s += dCtx[i][off + c] * V[j][off + c];
                        dV[j][off + c] += A[i][j] * dCtx[i][off + c];
                    }
                    dA[i][j] = s;
                }
            }
            // softmax backward per row
            for (int i = 0; i < n; i++) {
                float dot = 0;
                for (int k = 0; k < n; k++) dot += A[i][k] * dA[i][k];
                for (int j = 0; j < n; j++) {
                    float dS = A[i][j] * (dA[i][j] - dot) * scale;
                    for (int c = 0; c < headDim; c++) {
                        dQ[i][off + c] += dS * K[j][off + c];
                        dK[j][off + c] += dS * Q[i][off + c];
                    }
                }
            }
        }

        // Q, K, V = projections of the input
        float[][] dX = new float[n][dim];
        for (int t = 0; t < n; t++) {
            for (int d = 0; d < dim; d++) {
                bqg[d] += dQ[t][d];
                bkg[d] += dK[t][d];
                bvg[d] += dV[t][d];
                for (int j = 0; j < dim; j++) {
                    float x = lastInput[t][j];
                    Wqg[d][j] += dQ[t][d] * x;
                    Wkg[d][j] += dK[t][d] * x;
                    Wvg[d][j] += dV[t][d] * x;
                    dX[t][j] += Wq[d][j] * dQ[t][d] + Wk[d][j] * dK[t][d] + Wv[d][j] * dV[t][d];
                }
            }
        }
        return dX;
    }

    /** @return model dimension */
    public int dim() {
        return dim;
    }

    /** @return number of heads */
    public int heads() {
        return heads;
    }

    /** @return number of trainable parameters */
    public int parameterCount() {
        return 4 * dim * dim + 4 * dim;
    }

    /**
     * Replaces all weights and biases.
     *
     * @throws IllegalArgumentException on shape mismatch
     */
    public void loadParameters(float[][] wq, float[] bqv, float[][] wk, float[] bkv,
                               float[][] wv, float[] bvv, float[][] wo, float[] bov) {
        copyMatrix(wq, Wq, "Wq");
        copyMatrix(wk, Wk, "Wk");
        copyMatrix(wv, Wv, "Wv");
        copyMatrix(wo, Wo, "Wo");
        copyVector(bqv, bq, "bq");
        copyVector(bkv, bk, "bk");
        copyVector(bvv, bv, "bv");
        copyVector(bov, bo, "bo");
    }

    private void copyMatrix(float[][] src, float[][] dst, String name) {
        if (src.length != dim) throw new IllegalArgumentException(name + ": expected " + dim + " rows");
        for (float[] row : src) {
            if (row.length != dim) throw new IllegalArgumentException(name + ": expected rows of " + dim);
        }
        for (int i = 0; i < dim; i++) System.arraycopy(src[i], 0, dst[i], 0, dim);
    }

    private void copyVector(float[] src, float[] dst, String name) {
        if (src.length != dim) throw new IllegalArgumentException(name + ": expected length " + dim);
        System.arraycopy(src, 0, dst, 0, dim);
    }

    /** Copies Wq, Wk, Wv, Wo (row-major), then bq, bk, bv, bo. */
    public void copyParametersTo(float[] buf, int off) {
        copyMatrices(buf, off, false);
    }

    /** Inverse of {@link #copyParametersTo}. */
    public void copyParametersFrom(float[] buf, int off) {
        copyMatrices(buf, off, true);
    }

    /** Copies accumulated gradients in the same layout as {@link #copyParametersTo}. */
    public void copyGradientsTo(float[] buf, int off) {
        int p = off;
        for (float[][] m : new float[][][] {Wqg, Wkg, Wvg, Wog}) {
            for (float[] row : m) {
                System.arraycopy(row, 0, buf, p, dim);
                p += dim;
            }
        }
        for (float[] v : new float[][] {bqg, bkg, bvg, bog}) {
            System.arraycopy(v, 0, buf, p, dim);
            p += dim;
        }
    }

    private void copyMatrices(float[] buf, int off, boolean from) {
        int p = off;
        for (float[][] m : new float[][][] {Wq, Wk, Wv, Wo}) {
            for (float[] row : m) {
                if (from) System.arraycopy(buf, p, row, 0, dim);
                else System.arraycopy(row, 0, buf, p, dim);
                p += dim;
            }
        }
        for (float[] v : new float[][] {bq, bk, bv, bo}) {
            if (from) System.arraycopy(buf, p, v, 0, dim);
            else System.arraycopy(v, 0, buf, p, dim);
            p += dim;
        }
    }

    /** Clears accumulated gradients. */
    public void zeroGrad() {
        for (float[][] m : new float[][][] {Wqg, Wkg, Wvg, Wog}) {
            for (float[] row : m) Arrays.fill(row, 0.0f);
        }
        for (float[] v : new float[][] {bqg, bkg, bvg, bog}) Arrays.fill(v, 0.0f);
    }

    /** Applies accumulated gradients with SGD. */
    public void update(float learningRate, int batchSize) {
        float[] p = new float[parameterCount()];
        float[] g = new float[p.length];
        copyParametersTo(p, 0);
        copyGradientsTo(g, 0);
        float scale = -learningRate / batchSize;
        for (int i = 0; i < p.length; i++) p[i] += scale * g[i];
        copyParametersFrom(p, 0);
    }

    /**
     * Applies one step of an optimizer (for example {@link optim.Adam}) to all
     * parameters, using gradients averaged over the batch.
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
}
