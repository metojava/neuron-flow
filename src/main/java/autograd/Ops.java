package autograd;

import java.util.Arrays;

/**
 * Differentiable tensor operations for reverse-mode automatic
 * differentiation. Every method returns a new Tensor and, when
 * any input requires gradients, records a backward closure
 * that propagates gradients to the inputs.
 *
 * <p>Operations follow numpy broadcasting rules: dimensions are
 * aligned from the right, and missing leading dimensions are
 * treated as size 1.</p>
 */
public final class Ops {

    private Ops() {
    }

    // ── element-wise arithmetic ─────────────────────────────

    /**
     * Element-wise addition with numpy broadcasting.
     */
    public static Tensor add(Tensor a, Tensor b) {
        int[] outShape = broadcastShape(a.shape(), b.shape());
        int n = Tensor.numel(outShape);
        float[] out = new float[n];
        addKernels(out, outShape,
            a.data(), a.shape(),
            b.data(), b.shape());

        boolean track = a.requiresGrad() || b.requiresGrad();
        Tensor r = new Tensor(out, outShape, track);
        if (track) {
            r.setBackward(new Tensor[]{a, b}, () -> {
                float[] g = r.grad();
                if (a.requiresGrad()) {
                    reduceInto(a.grad(), g, outShape, a.shape());
                }
                if (b.requiresGrad()) {
                    reduceInto(b.grad(), g, outShape, b.shape());
                }
            });
        }
        return r;
    }

    /**
     * Element-wise subtraction with numpy broadcasting.
     * Computes {@code a - b}.
     */
    public static Tensor sub(Tensor a, Tensor b) {
        int[] outShape = broadcastShape(a.shape(), b.shape());
        int n = Tensor.numel(outShape);
        float[] out = new float[n];
        subKernels(out, outShape,
            a.data(), a.shape(),
            b.data(), b.shape());

        boolean track = a.requiresGrad() || b.requiresGrad();
        Tensor r = new Tensor(out, outShape, track);
        if (track) {
            r.setBackward(new Tensor[]{a, b}, () -> {
                float[] g = r.grad();
                if (a.requiresGrad()) {
                    reduceInto(a.grad(), g, outShape, a.shape());
                }
                if (b.requiresGrad()) {
                    float[] bg = b.grad();
                    reduceIntoNeg(bg, g, outShape, b.shape());
                }
            });
        }
        return r;
    }

    /**
     * Element-wise multiplication with numpy broadcasting.
     */
    public static Tensor mul(Tensor a, Tensor b) {
        int[] outShape = broadcastShape(a.shape(), b.shape());
        int n = Tensor.numel(outShape);
        float[] out = new float[n];
        mulKernels(out, outShape,
            a.data(), a.shape(),
            b.data(), b.shape());

        boolean track = a.requiresGrad() || b.requiresGrad();
        Tensor r = new Tensor(out, outShape, track);
        if (track) {
            // save copies of inputs for backward
            float[] aData = a.requiresGrad() ? a.data().clone() : null;
            float[] bData = b.requiresGrad() ? b.data().clone() : null;
            int[] aSh = a.shape(), bSh = b.shape();
            r.setBackward(new Tensor[]{a, b}, () -> {
                float[] g = r.grad();
                if (a.requiresGrad()) {
                    // da = g * b
                    float[] tmp = new float[n];
                    mulKernels(tmp, outShape,
                        g, outShape, bData, bSh);
                    reduceInto(a.grad(), tmp, outShape, aSh);
                }
                if (b.requiresGrad()) {
                    // db = g * a
                    float[] tmp = new float[n];
                    mulKernels(tmp, outShape,
                        g, outShape, aData, aSh);
                    reduceInto(b.grad(), tmp, outShape, bSh);
                }
            });
        }
        return r;
    }

    /**
     * Multiplies every element by a constant scalar.
     */
    public static Tensor scale(Tensor x, float s) {
        int[] sh = x.shape();
        int n = x.numel();
        float[] out = new float[n];
        float[] xd = x.data();
        for (int i = 0; i < n; i++) out[i] = xd[i] * s;

        Tensor r = new Tensor(out, sh, x.requiresGrad());
        if (x.requiresGrad()) {
            r.setBackward(new Tensor[]{x}, () -> {
                float[] g = r.grad();
                float[] xg = x.grad();
                for (int i = 0; i < n; i++) xg[i] += g[i] * s;
            });
        }
        return r;
    }

    /**
     * Adds a constant scalar to every element.
     */
    public static Tensor addScalar(Tensor x, float s) {
        int[] sh = x.shape();
        int n = x.numel();
        float[] out = new float[n];
        float[] xd = x.data();
        for (int i = 0; i < n; i++) out[i] = xd[i] + s;

        Tensor r = new Tensor(out, sh, x.requiresGrad());
        if (x.requiresGrad()) {
            r.setBackward(new Tensor[]{x}, () -> {
                float[] g = r.grad();
                float[] xg = x.grad();
                for (int i = 0; i < n; i++) xg[i] += g[i];
            });
        }
        return r;
    }

    // ── matrix multiplication ───────────────────────────────

    /**
     * Batched matrix multiply. The last two dimensions of each
     * operand are the matrix dimensions; all leading dimensions
     * are batch dimensions.
     *
     * <p>Cases:</p>
     * <ul>
     *   <li>{M,K} x {K,N} -&gt; {M,N}</li>
     *   <li>{B,M,K} x {K,N} -&gt; {B,M,N} (2D b broadcast)</li>
     *   <li>{B,M,K} x {B,K,N} -&gt; {B,M,N}</li>
     *   <li>{B1,B2,M,K} x {B1,B2,K,N} -&gt; {B1,B2,M,N}</li>
     * </ul>
     */
    public static Tensor matmul(Tensor a, Tensor b) {
        int[] as = a.shape(), bs = b.shape();
        if (as.length < 2 || bs.length < 2) {
            throw new IllegalArgumentException(
                "matmul requires at least 2D tensors, got "
                + as.length + "D and " + bs.length + "D");
        }
        int K = as[as.length - 1];
        int Kb = bs[bs.length - 2];
        if (K != Kb) {
            throw new IllegalArgumentException(
                "matmul inner dims mismatch: " + K + " vs " + Kb);
        }

        int M = as[as.length - 2];
        int N = bs[bs.length - 1];

        // batch shape = a's leading dims (b must match or be 2D)
        int[] aBatch = Arrays.copyOf(as, as.length - 2);
        int[] bBatch = Arrays.copyOf(bs, bs.length - 2);
        boolean bBroadcast = bBatch.length == 0
            && aBatch.length > 0;
        int batchCount = Tensor.numel(aBatch);

        int[] outShape = new int[as.length];
        System.arraycopy(aBatch, 0, outShape, 0, aBatch.length);
        outShape[as.length - 2] = M;
        outShape[as.length - 1] = N;

        float[] out = new float[Tensor.numel(outShape)];
        float[] ad = a.data(), bd = b.data();

        for (int batch = 0; batch < batchCount; batch++) {
            int aOff = batch * M * K;
            int bOff = bBroadcast ? 0 : batch * K * N;
            int oOff = batch * M * N;
            matmulKernel(ad, aOff, bd, bOff,
                out, oOff, M, K, N);
        }

        boolean track = a.requiresGrad() || b.requiresGrad();
        Tensor r = new Tensor(out, outShape, track);
        if (track) {
            r.setBackward(new Tensor[]{a, b}, () -> {
                float[] g = r.grad();
                if (a.requiresGrad()) {
                    float[] ag = a.grad();
                    // da = dout @ b^T
                    for (int batch = 0;
                         batch < batchCount; batch++) {
                        int gOff = batch * M * N;
                        int bOff = bBroadcast ? 0 : batch * K * N;
                        int aOff = batch * M * K;
                        matmulBT(g, gOff, bd, bOff,
                            ag, aOff, M, N, K);
                    }
                }
                if (b.requiresGrad()) {
                    float[] bg = b.grad();
                    if (bBroadcast) {
                        // db = sum_batches a^T @ dout
                        for (int batch = 0;
                             batch < batchCount; batch++) {
                            int gOff = batch * M * N;
                            int aOff = batch * M * K;
                            matmulATB(ad, aOff, g, gOff,
                                bg, 0, K, M, N, true);
                        }
                    } else {
                        // db = a^T @ dout (per batch)
                        for (int batch = 0;
                             batch < batchCount; batch++) {
                            int gOff = batch * M * N;
                            int aOff = batch * M * K;
                            int bOff = batch * K * N;
                            matmulATB(ad, aOff, g, gOff,
                                bg, bOff, K, M, N, true);
                        }
                    }
                }
            });
        }
        return r;
    }

    // ── activations ─────────────────────────────────────────

    /**
     * Rectified linear unit: max(0, x).
     */
    public static Tensor relu(Tensor x) {
        int n = x.numel();
        float[] xd = x.data();
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = xd[i] > 0 ? xd[i] : 0.0f;
        }

        Tensor r = new Tensor(out, x.shape(),
            x.requiresGrad());
        if (x.requiresGrad()) {
            r.setBackward(new Tensor[]{x}, () -> {
                float[] g = r.grad();
                float[] xg = x.grad();
                for (int i = 0; i < n; i++) {
                    if (xd[i] > 0) xg[i] += g[i];
                }
            });
        }
        return r;
    }

    /**
     * Gaussian Error Linear Unit using the tanh approximation:
     * 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))).
     */
    public static Tensor gelu(Tensor x) {
        int n = x.numel();
        float[] xd = x.data();
        float[] out = new float[n];
        float c = (float) Math.sqrt(2.0 / Math.PI);
        for (int i = 0; i < n; i++) {
            float v = xd[i];
            float u = c * (v + 0.044715f * v * v * v);
            out[i] = 0.5f * v * (1.0f + (float) Math.tanh(u));
        }

        Tensor r = new Tensor(out, x.shape(),
            x.requiresGrad());
        if (x.requiresGrad()) {
            r.setBackward(new Tensor[]{x}, () -> {
                float[] g = r.grad();
                float[] xg = x.grad();
                for (int i = 0; i < n; i++) {
                    float v = xd[i];
                    float u = c * (v + 0.044715f * v * v * v);
                    float t = (float) Math.tanh(u);
                    float du = c * (1.0f
                        + 3.0f * 0.044715f * v * v);
                    float dgelu = 0.5f * (1.0f + t)
                        + 0.5f * v * (1.0f - t * t) * du;
                    xg[i] += g[i] * dgelu;
                }
            });
        }
        return r;
    }

    /**
     * Hyperbolic tangent activation.
     */
    public static Tensor tanh(Tensor x) {
        int n = x.numel();
        float[] xd = x.data();
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) Math.tanh(xd[i]);
        }

        Tensor r = new Tensor(out, x.shape(),
            x.requiresGrad());
        if (x.requiresGrad()) {
            r.setBackward(new Tensor[]{x}, () -> {
                float[] g = r.grad();
                float[] xg = x.grad();
                for (int i = 0; i < n; i++) {
                    float t = out[i];
                    xg[i] += g[i] * (1.0f - t * t);
                }
            });
        }
        return r;
    }

    // ── softmax & loss ──────────────────────────────────────

    /**
     * Softmax over the last dimension.
     */
    public static Tensor softmax(Tensor x) {
        int last = x.ndim() - 1;
        int outer = 1;
        for (int i = 0; i < last; i++) outer *= x.size(i);
        final int outerF = outer;
        final int VF = x.size(last);
        float[] xd = x.data();
        float[] out = new float[xd.length];

        for (int o = 0; o < outerF; o++) {
            int off = o * VF;
            float max = xd[off];
            for (int j = 1; j < VF; j++) {
                if (xd[off + j] > max) max = xd[off + j];
            }
            float sum = 0;
            for (int j = 0; j < VF; j++) {
                out[off + j] = (float) Math.exp(xd[off + j] - max);
                sum += out[off + j];
            }
            float inv = 1.0f / sum;
            for (int j = 0; j < VF; j++) out[off + j] *= inv;
        }

        Tensor r = new Tensor(out, x.shape(),
            x.requiresGrad());
        if (x.requiresGrad()) {
            r.setBackward(new Tensor[]{x}, () -> {
                float[] g = r.grad();
                float[] xg = x.grad();
                // dx_i = y_i * (dy_i - sum_j dy_j * y_j)
                for (int o = 0; o < outerF; o++) {
                    int off = o * VF;
                    float dot = 0;
                    for (int j = 0; j < VF; j++) {
                        dot += g[off + j] * out[off + j];
                    }
                    for (int j = 0; j < VF; j++) {
                        float y = out[off + j];
                        xg[off + j] += y * (g[off + j] - dot);
                    }
                }
            });
        }
        return r;
    }

    /**
     * Fused softmax + cross-entropy loss. Accepts logits of
     * shape {N, V} or {B, T, V} (flattened internally) and
     * integer targets of length N = prod(shape) / V.
     * Returns a scalar tensor (mean loss).
     *
     * @param logits  unnormalised scores, last dim = vocab
     * @param targets true class index for each row
     * @return scalar mean cross-entropy
     */
    public static Tensor crossEntropy(Tensor logits,
                                      int[] targets) {
        int V = logits.size(logits.ndim() - 1);
        int N = logits.numel() / V;
        if (targets.length != N) {
            throw new IllegalArgumentException(
                "targets length " + targets.length
                + " != rows " + N);
        }
        float[] ld = logits.data();

        float[] probs = new float[ld.length];
        float lossSum = 0;
        for (int i = 0; i < N; i++) {
            int off = i * V;
            float max = ld[off];
            for (int j = 1; j < V; j++) {
                if (ld[off + j] > max) max = ld[off + j];
            }
            float sum = 0;
            for (int j = 0; j < V; j++) {
                probs[off + j] =
                    (float) Math.exp(ld[off + j] - max);
                sum += probs[off + j];
            }
            float inv = 1.0f / sum;
            for (int j = 0; j < V; j++) probs[off + j] *= inv;
            int t = targets[i];
            lossSum += -(float) Math.log(
                Math.max(probs[off + t], 1e-12f));
        }
        float meanLoss = lossSum / N;

        Tensor r = new Tensor(
            new float[]{meanLoss}, new int[]{1},
            logits.requiresGrad());
        if (logits.requiresGrad()) {
            int[] logShape = logits.shape();
            r.setBackward(new Tensor[]{logits}, () -> {
                float[] lg = logits.grad();
                float scale = 1.0f / N;
                for (int i = 0; i < N; i++) {
                    int off = i * V;
                    int t = targets[i];
                    for (int j = 0; j < V; j++) {
                        float grad = probs[off + j] * scale;
                        if (j == t) grad -= scale;
                        lg[off + j] += grad;
                    }
                }
            });
        }
        return r;
    }

    // ── layer normalisation ─────────────────────────────────

    /**
     * Layer normalisation over the last dimension:
     * out = gamma * (x - mean) / sqrt(var + eps) + beta.
     *
     * @param x     input of shape {..., C}
     * @param gamma per-channel scale, shape {C}, trainable
     * @param beta  per-channel shift, shape {C}, trainable
     * @return normalised tensor, same shape as x
     */
    public static Tensor layerNorm(Tensor x, Tensor gamma,
                                   Tensor beta) {
        int C = x.size(x.ndim() - 1);
        int outer = x.numel() / C;
        float eps = 1e-5f;
        float[] xd = x.data();
        float[] gd = gamma.data();
        float[] bd = beta.data();
        float[] out = new float[xd.length];
        float[] xhat = new float[xd.length];

        for (int o = 0; o < outer; o++) {
            int off = o * C;
            float mean = 0;
            for (int j = 0; j < C; j++) mean += xd[off + j];
            mean /= C;
            float var = 0;
            for (int j = 0; j < C; j++) {
                float d = xd[off + j] - mean;
                var += d * d;
            }
            var /= C;
            float invStd = 1.0f / (float) Math.sqrt(var + eps);
            for (int j = 0; j < C; j++) {
                float h = (xd[off + j] - mean) * invStd;
                xhat[off + j] = h;
                out[off + j] = gd[j] * h + bd[j];
            }
        }

        boolean track = x.requiresGrad()
            || gamma.requiresGrad() || beta.requiresGrad();
        Tensor r = new Tensor(out, x.shape(), track);
        if (track) {
            r.setBackward(new Tensor[]{x, gamma, beta}, () -> {
                float[] g = r.grad();
                if (gamma.requiresGrad()) {
                    float[] gg = gamma.grad();
                    for (int o = 0; o < outer; o++) {
                        int off = o * C;
                        for (int j = 0; j < C; j++) {
                            gg[j] += g[off + j] * xhat[off + j];
                        }
                    }
                }
                if (beta.requiresGrad()) {
                    float[] bg = beta.grad();
                    for (int o = 0; o < outer; o++) {
                        int off = o * C;
                        for (int j = 0; j < C; j++) {
                            bg[j] += g[off + j];
                        }
                    }
                }
                if (x.requiresGrad()) {
                    float[] xg = x.grad();
                    float[] gEd = new float[C];
                    for (int o = 0; o < outer; o++) {
                        int off = o * C;
                        // dx_hat = g * gamma
                        for (int j = 0; j < C; j++) {
                            gEd[j] = g[off + j] * gd[j];
                        }
                        // mean of dx_hat
                        float m1 = 0;
                        for (int j = 0; j < C; j++) m1 += gEd[j];
                        m1 /= C;
                        // mean of dx_hat * xhat
                        float m2 = 0;
                        for (int j = 0; j < C; j++) {
                            m2 += gEd[j] * xhat[off + j];
                        }
                        m2 /= C;
                        // recompute invStd
                        float mean = 0;
                        for (int j = 0; j < C; j++) {
                            mean += xd[off + j];
                        }
                        mean /= C;
                        float var = 0;
                        for (int j = 0; j < C; j++) {
                            float d = xd[off + j] - mean;
                            var += d * d;
                        }
                        var /= C;
                        float invStd = 1.0f
                            / (float) Math.sqrt(var + eps);
                        for (int j = 0; j < C; j++) {
                            xg[off + j] += invStd * (gEd[j] - m1
                                - xhat[off + j] * m2);
                        }
                    }
                }
            });
        }
        return r;
    }

    // ── embedding ───────────────────────────────────────────

    /**
     * Looks up rows of a weight matrix by integer index.
     *
     * @param weight  shape {vocab, D}
     * @param indices token ids, length N
     * @param outShape output shape (typically {B, T, D})
     * @return tensor of shape outShape
     */
    public static Tensor embedding(Tensor weight,
                                   int[] indices,
                                   int... outShape) {
        int D = weight.size(1);
        int expected = 1;
        for (int s : outShape) expected *= s;
        if (expected != indices.length * D) {
            throw new IllegalArgumentException(
                "indices length " + indices.length
                + " * D=" + D + " != product of outShape "
                + expected);
        }
        float[] wd = weight.data();
        float[] out = new float[indices.length * D];
        for (int i = 0; i < indices.length; i++) {
            System.arraycopy(wd, indices[i] * D, out,
                i * D, D);
        }

        Tensor r = new Tensor(out, outShape,
            weight.requiresGrad());
        if (weight.requiresGrad()) {
            r.setBackward(new Tensor[]{weight}, () -> {
                float[] g = r.grad();
                float[] wg = weight.grad();
                for (int i = 0; i < indices.length; i++) {
                    int wOff = indices[i] * D;
                    int gOff = i * D;
                    for (int j = 0; j < D; j++) {
                        wg[wOff + j] += g[gOff + j];
                    }
                }
            });
        }
        return r;
    }

    // ── shape operations ────────────────────────────────────

    /**
     * Reshapes a tensor. Data layout is unchanged (row-major),
     * so backward is a flat accumulation.
     */
    public static Tensor reshape(Tensor x, int... newShape) {
        int oldN = x.numel();
        int newN = Tensor.numel(newShape);
        if (oldN != newN) {
            throw new IllegalArgumentException(
                "cannot reshape " + oldN + " elements to "
                + Arrays.toString(newShape) + " ("
                + newN + " elements)");
        }
        float[] out = new float[oldN];
        System.arraycopy(x.data(), 0, out, 0, oldN);

        Tensor r = new Tensor(out, newShape.clone(),
            x.requiresGrad());
        if (x.requiresGrad()) {
            r.setBackward(new Tensor[]{x}, () -> {
                float[] g = r.grad();
                float[] xg = x.grad();
                for (int i = 0; i < oldN; i++) {
                    xg[i] += g[i];
                }
            });
        }
        return r;
    }

    /**
     * Flattens all dimensions except the last:
     * {a, b, c} -&gt; {a*b, c}. Convenience for linear layers.
     */
    public static Tensor flattenLeading(Tensor x) {
        int last = x.size(x.ndim() - 1);
        int rows = x.numel() / last;
        return reshape(x, rows, last);
    }

    /**
     * Restores a tensor flattened by {@link #flattenLeading}
     * back to the original leading dimensions.
     */
    public static Tensor unflattenLeading(Tensor x,
                                          int... leading) {
        int last = x.size(1);
        int[] sh = new int[leading.length + 1];
        System.arraycopy(leading, 0, sh, 0, leading.length);
        sh[leading.length] = last;
        return reshape(x, sh);
    }

    /**
     * Swaps two dimensions of a tensor.
     *
     * @param x  input tensor
     * @param d1 first dimension to swap (negative OK)
     * @param d2 second dimension to swap (negative OK)
     * @return transposed tensor
     */
    public static Tensor transpose(Tensor x, int d1, int d2) {
        int nd = x.ndim();
        if (d1 < 0) d1 += nd;
        if (d2 < 0) d2 += nd;
        if (d1 == d2 || d1 < 0 || d1 >= nd
                || d2 < 0 || d2 >= nd) {
            throw new IllegalArgumentException(
                "invalid transpose dims " + d1 + ", " + d2
                + " for " + nd + "D tensor");
        }
        final int s1 = d1, s2 = d2;
        int[] inSh = x.shape();
        int[] outSh = inSh.clone();
        int tmp = outSh[s1];
        outSh[s1] = outSh[s2];
        outSh[s2] = tmp;

        float[] in = x.data();
        float[] out = new float[in.length];

        // iterate over all output positions
        final int total = in.length;
        final int ndF = nd;
        int[] inIdx = new int[nd];
        int[] outIdx = new int[nd];
        for (int flat = 0; flat < total; flat++) {
            // decompose flat into outIdx
            int rem = flat;
            for (int d = nd - 1; d >= 0; d--) {
                outIdx[d] = rem % outSh[d];
                rem /= outSh[d];
            }
            // map to inIdx (swap back the two dims)
            System.arraycopy(outIdx, 0, inIdx, 0, nd);
            int t = inIdx[s1];
            inIdx[s1] = inIdx[s2];
            inIdx[s2] = t;
            // compute input offset
            int iOff = 0;
            for (int d = 0; d < nd; d++) {
                iOff = iOff * inSh[d] + inIdx[d];
            }
            out[flat] = in[iOff];
        }

        Tensor r = new Tensor(out, outSh,
            x.requiresGrad());
        if (x.requiresGrad()) {
            // backward: transpose grad back (swap d1, d2 again)
            r.setBackward(new Tensor[]{x}, () -> {
                float[] g = r.grad();
                float[] xg = x.grad();
                for (int flat = 0; flat < total; flat++) {
                    int rem = flat;
                    for (int d = ndF - 1; d >= 0; d--) {
                        outIdx[d] = rem % outSh[d];
                        rem /= outSh[d];
                    }
                    System.arraycopy(outIdx, 0, inIdx, 0, ndF);
                    int t = inIdx[s1];
                    inIdx[s1] = inIdx[s2];
                    inIdx[s2] = t;
                    int iOff = 0;
                    for (int d = 0; d < ndF; d++) {
                        iOff = iOff * inSh[d] + inIdx[d];
                    }
                    xg[iOff] += g[flat];
                }
            });
        }
        return r;
    }

    // ── masks ───────────────────────────────────────────────

    /**
     * Creates a causal attention mask of shape {T, T}:
     * 0 where column &lt;= row (attend to past + self) and
     * -1e9 where column &gt; row (attend to future). Constant
     * tensor, no gradients.
     */
    public static Tensor causalMask(int T) {
        float[] out = new float[T * T];
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                if (j > i) out[i * T + j] = -1e9f;
            }
        }
        return new Tensor(out, new int[]{T, T}, false);
    }

    // ── gather ──────────────────────────────────────────────

    /**
     * Selects rows from a 2D tensor by index.
     *
     * @param x       2D tensor {N, V}
     * @param indices row indices to extract, length M
     * @return tensor {M, V}
     */
    public static Tensor gatherRows(Tensor x, int[] indices) {
        if (x.ndim() != 2) {
            throw new IllegalArgumentException(
                "gatherRows requires 2D tensor, got "
                + x.ndim() + "D");
        }
        int N = x.size(0), V = x.size(1);
        float[] xd = x.data();
        float[] out = new float[indices.length * V];
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= N) {
                throw new IllegalArgumentException(
                    "row index " + indices[i]
                    + " out of bounds [0," + N + ")");
            }
            System.arraycopy(xd, indices[i] * V, out,
                i * V, V);
        }

        Tensor r = new Tensor(out,
            new int[]{indices.length, V},
            x.requiresGrad());
        if (x.requiresGrad()) {
            r.setBackward(new Tensor[]{x}, () -> {
                float[] g = r.grad();
                float[] xg = x.grad();
                for (int i = 0; i < indices.length; i++) {
                    int sOff = indices[i] * V;
                    int gOff = i * V;
                    for (int j = 0; j < V; j++) {
                        xg[sOff + j] += g[gOff + j];
                    }
                }
            });
        }
        return r;
    }

    // ── broadcast helpers ───────────────────────────────────

    static int[] broadcastShape(int[] a, int[] b) {
        int nd = Math.max(a.length, b.length);
        int[] out = new int[nd];
        for (int i = 0; i < nd; i++) {
            int da = i < nd - a.length ? 1
                : a[i - (nd - a.length)];
            int db = i < nd - b.length ? 1
                : b[i - (nd - b.length)];
            if (da != db && da != 1 && db != 1) {
                throw new IllegalArgumentException(
                    "cannot broadcast shapes "
                    + Arrays.toString(a) + " and "
                    + Arrays.toString(b));
            }
            out[i] = Math.max(da, db);
        }
        return out;
    }

    private static void reduceInto(float[] dst, float[] src,
                                   int[] srcShape,
                                   int[] dstShape) {
        if (Arrays.equals(srcShape, dstShape)) {
            for (int i = 0; i < src.length; i++) {
                dst[i] += src[i];
            }
            return;
        }
        // generic broadcast reduction
        int srcN = src.length;
        int dstN = dstShape.length;
        int[] fullSrc = new int[dstN];
        int[] fullDst = new int[dstN];
        // pad srcShape on the left with 1s
        int pad = dstN - srcShape.length;
        for (int i = 0; i < dstN; i++) {
            fullSrc[i] = i < pad ? 1 : srcShape[i - pad];
            fullDst[i] = dstShape[i];
        }
        int[] srcIdx = new int[dstN];
        for (int flat = 0; flat < srcN; flat++) {
            // decompose flat into srcIdx (over fullSrc)
            int rem = flat;
            for (int d = dstN - 1; d >= 0; d--) {
                srcIdx[d] = rem % fullSrc[d];
                rem /= fullSrc[d];
            }
            // map to dst offset (where src dim is 1, use 0)
            int dOff = 0;
            for (int d = 0; d < dstN; d++) {
                int idx = fullSrc[d] == 1 ? 0 : srcIdx[d];
                dOff = dOff * fullDst[d] + idx;
            }
            dst[dOff] += src[flat];
        }
    }

    private static void reduceIntoNeg(float[] dst, float[] src,
                                      int[] srcShape,
                                      int[] dstShape) {
        if (Arrays.equals(srcShape, dstShape)) {
            for (int i = 0; i < src.length; i++) {
                dst[i] -= src[i];
            }
            return;
        }
        int srcN = src.length;
        int dstN = dstShape.length;
        int[] fullSrc = new int[dstN];
        int[] fullDst = new int[dstN];
        int pad = dstN - srcShape.length;
        for (int i = 0; i < dstN; i++) {
            fullSrc[i] = i < pad ? 1 : srcShape[i - pad];
            fullDst[i] = dstShape[i];
        }
        int[] srcIdx = new int[dstN];
        for (int flat = 0; flat < srcN; flat++) {
            int rem = flat;
            for (int d = dstN - 1; d >= 0; d--) {
                srcIdx[d] = rem % fullSrc[d];
                rem /= fullSrc[d];
            }
            int dOff = 0;
            for (int d = 0; d < dstN; d++) {
                int idx = fullSrc[d] == 1 ? 0 : srcIdx[d];
                dOff = dOff * fullDst[d] + idx;
            }
            dst[dOff] -= src[flat];
        }
    }

    // ── element-wise kernels (broadcasting) ─────────────────

    private static void addKernels(float[] out, int[] outSh,
                                   float[] a, int[] aSh,
                                   float[] b, int[] bSh) {
        int n = out.length;
        int nd = outSh.length;
        int[] aFull = new int[nd];
        int[] bFull = new int[nd];
        int padA = nd - aSh.length, padB = nd - bSh.length;
        for (int i = 0; i < nd; i++) {
            aFull[i] = i < padA ? 1 : aSh[i - padA];
            bFull[i] = i < padB ? 1 : bSh[i - padB];
        }
        int[] idx = new int[nd];
        for (int flat = 0; flat < n; flat++) {
            int rem = flat, aOff = 0, bOff = 0;
            for (int d = nd - 1; d >= 0; d--) {
                idx[d] = rem % outSh[d];
                rem /= outSh[d];
            }
            for (int d = 0; d < nd; d++) {
                aOff = aOff * aFull[d]
                    + (aFull[d] == 1 ? 0 : idx[d]);
                bOff = bOff * bFull[d]
                    + (bFull[d] == 1 ? 0 : idx[d]);
            }
            out[flat] = a[aOff] + b[bOff];
        }
    }

    private static void subKernels(float[] out, int[] outSh,
                                   float[] a, int[] aSh,
                                   float[] b, int[] bSh) {
        int n = out.length;
        int nd = outSh.length;
        int[] aFull = new int[nd];
        int[] bFull = new int[nd];
        int padA = nd - aSh.length, padB = nd - bSh.length;
        for (int i = 0; i < nd; i++) {
            aFull[i] = i < padA ? 1 : aSh[i - padA];
            bFull[i] = i < padB ? 1 : bSh[i - padB];
        }
        int[] idx = new int[nd];
        for (int flat = 0; flat < n; flat++) {
            int rem = flat, aOff = 0, bOff = 0;
            for (int d = nd - 1; d >= 0; d--) {
                idx[d] = rem % outSh[d];
                rem /= outSh[d];
            }
            for (int d = 0; d < nd; d++) {
                aOff = aOff * aFull[d]
                    + (aFull[d] == 1 ? 0 : idx[d]);
                bOff = bOff * bFull[d]
                    + (bFull[d] == 1 ? 0 : idx[d]);
            }
            out[flat] = a[aOff] - b[bOff];
        }
    }

    private static void mulKernels(float[] out, int[] outSh,
                                   float[] a, int[] aSh,
                                   float[] b, int[] bSh) {
        int n = out.length;
        int nd = outSh.length;
        int[] aFull = new int[nd];
        int[] bFull = new int[nd];
        int padA = nd - aSh.length, padB = nd - bSh.length;
        for (int i = 0; i < nd; i++) {
            aFull[i] = i < padA ? 1 : aSh[i - padA];
            bFull[i] = i < padB ? 1 : bSh[i - padB];
        }
        int[] idx = new int[nd];
        for (int flat = 0; flat < n; flat++) {
            int rem = flat, aOff = 0, bOff = 0;
            for (int d = nd - 1; d >= 0; d--) {
                idx[d] = rem % outSh[d];
                rem /= outSh[d];
            }
            for (int d = 0; d < nd; d++) {
                aOff = aOff * aFull[d]
                    + (aFull[d] == 1 ? 0 : idx[d]);
                bOff = bOff * bFull[d]
                    + (bFull[d] == 1 ? 0 : idx[d]);
            }
            out[flat] = a[aOff] * b[bOff];
        }
    }

    // ── matmul kernels ──────────────────────────────────────

    private static void matmulKernel(
            float[] a, int aOff, float[] b, int bOff,
            float[] out, int oOff, int M, int K, int N) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0;
                int aRow = aOff + m * K;
                for (int k = 0; k < K; k++) {
                    sum += a[aRow + k]
                        * b[bOff + k * N + n];
                }
                out[oOff + m * N + n] = sum;
            }
        }
    }

    // C = A @ B^T: A is MxK, B is NxK, C is MxN
    private static void matmulBT(
            float[] a, int aOff, float[] b, int bOff,
            float[] out, int oOff, int M, int K, int N) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0;
                int aRow = aOff + m * K;
                int bRow = bOff + n * K;
                for (int k = 0; k < K; k++) {
                    sum += a[aRow + k] * b[bRow + k];
                }
                out[oOff + m * N + n] = sum;
            }
        }
    }

    // C += A^T @ B: A is MxK, B is MxN, C is KxN
    private static void matmulATB(
            float[] a, int aOff, float[] b, int bOff,
            float[] c, int cOff, int K, int M, int N,
            boolean accumulate) {
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                float sum = 0;
                for (int m = 0; m < M; m++) {
                    sum += a[aOff + m * K + k]
                        * b[bOff + m * N + n];
                }
                int idx = cOff + k * N + n;
                c[idx] = accumulate ? c[idx] + sum : sum;
            }
        }
    }
}
