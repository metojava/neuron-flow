package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Multi-head scaled dot-product attention with optional RoPE.
 *
 * <p>For input of shape {B, T, D}, we project to queries,
 * keys, values, split into H heads, compute attention, then
 * concatenate back to {B, T, D}.</p>
 */
public final class MultiHeadAttention {

    private final Linear qProj;
    private final Linear kProj;
    private final Linear vProj;
    private final Linear outProj;
    private final int dModel;
    private final int numHeads;
    private final int headDim;
    private final boolean useRoPE;
    private final Tensor ropeCos;
    private final Tensor ropeSin;
    private final Dropout attnDropout;
    private final Dropout projDropout;

    public MultiHeadAttention(int dModel, int numHeads,
                              long seed) {
        this(dModel, numHeads, seed, false, 2048, 0.1f, 0.1f);
    }

    /**
     * Creates attention with optional Rotary Positional Embeddings (RoPE).
     *
     * @param dModel      model dimension
     * @param numHeads    number of attention heads
     * @param seed        RNG seed
     * @param useRoPE     whether to apply RoPE
     * @param maxSeqLen   maximum sequence length for RoPE precomputation
     * @param attnDropout attention weight dropout rate
     * @param projDropout output projection dropout rate
     */
    public MultiHeadAttention(int dModel, int numHeads,
                              long seed, boolean useRoPE, int maxSeqLen,
                              float attnDropoutRate, float projDropoutRate) {
        if (dModel <= 0 || numHeads <= 0
                || dModel % numHeads != 0) {
            throw new IllegalArgumentException(
                "dModel must be divisible by numHeads, got dModel="
                + dModel + ", numHeads=" + numHeads);
        }
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;
        this.useRoPE = useRoPE;
        this.qProj = new Linear(dModel, dModel, seed + 1);
        this.kProj = new Linear(dModel, dModel, seed + 2);
        this.vProj = new Linear(dModel, dModel, seed + 3);
        this.outProj = new Linear(dModel, dModel, seed + 4);
        this.attnDropout = new Dropout(attnDropoutRate, seed + 100);
        this.projDropout = new Dropout(projDropoutRate, seed + 101);

        if (useRoPE) {
            float[] cos = new float[maxSeqLen * headDim];
            float[] sin = new float[maxSeqLen * headDim];
            for (int pos = 0; pos < maxSeqLen; pos++) {
                for (int i = 0; i < headDim; i += 2) {
                    float theta = (float) (pos * Math.pow(10000.0, -i / (float) headDim));
                    cos[pos * headDim + i] = (float) Math.cos(theta);
                    cos[pos * headDim + i + 1] = (float) Math.cos(theta);
                    sin[pos * headDim + i] = (float) Math.sin(theta);
                    sin[pos * headDim + i + 1] = (float) Math.sin(theta);
                }
            }
            this.ropeCos = Tensor.of(cos, maxSeqLen, headDim);
            this.ropeSin = Tensor.of(sin, maxSeqLen, headDim);
        } else {
            this.ropeCos = null;
            this.ropeSin = null;
        }
    }

    /**
     * Forward pass with causal mask (for GPT) or optional mask.
     *
     * @param x           input of shape {B, T, D}
     * @param useCausal   mask future tokens
     * @param keyPaddingMask boolean mask of shape {B, T} where true = mask this position (padding)
     * @return attention output of shape {B, T, D}
     */
    public Tensor forward(Tensor x, boolean useCausal, boolean[][] keyPaddingMask) {
        return forward(x, useCausal, null, null, 0, keyPaddingMask)[0];
    }

    /**
     * Forward pass with causal mask (for GPT) or optional mask.
     *
     * @param x           input of shape {B, T, D}
     * @param useCausal   mask future tokens
     * @return attention output of shape {B, T, D}
     */
    public Tensor forward(Tensor x, boolean useCausal) {
        return forward(x, useCausal, null, null, 0, null)[0];
    }

    /**
     * Forward pass with optional KV cache for autoregressive generation.
     *
     * @param x           input of shape {B, T, D} (typically T=1 for generation)
     * @param useCausal   apply causal mask
     * @param pastK       cached keys of shape {B, H, pastT, dH} or null
     * @param pastV       cached values of shape {B, H, pastT, dH} or null
     * @param pastLen     length of cached sequence (pastT)
     * @param keyPaddingMask boolean mask of shape {B, T} where true = mask this position (padding)
     * @return tuple: [output {B, T, D}, newK {B, H, T+pastT, dH}, newV {B, H, T+pastT, dH}]
     */
    public Tensor[] forward(Tensor x, boolean useCausal,
                            Tensor pastK, Tensor pastV, int pastLen,
                            boolean[][] keyPaddingMask) {
        if (x.ndim() != 3) {
            throw new IllegalArgumentException(
                "expected 3D {B, T, D}, got " + x.ndim() + "D");
        }
        if (x.size(2) != dModel) {
            throw new IllegalArgumentException(
                "expected D=" + dModel + ", got " + x.size(2));
        }

        int B = x.size(0), T = x.size(1);
        Tensor Q = qProj.forward(x); // {B, T, D}
        Tensor K = kProj.forward(x); // {B, T, D}
        Tensor V = vProj.forward(x); // {B, T, D}

        Q = reshapeHeads(Q, B, T);
        K = reshapeHeads(K, B, T);
        V = reshapeHeads(V, B, T); // {B, H, T, dH}

        // Apply RoPE if enabled
        if (useRoPE) {
            Q = applyRoPE(Q, pastLen);
            K = applyRoPE(K, pastLen);
        }

        // Append to cache if provided
        if (pastK != null && pastV != null) {
            K = concatCache(pastK, K); // {B, H, pastT+T, dH}
            V = concatCache(pastV, V);
        }

        int curT = K.size(2);

        // Q @ K^T / sqrt(dH)
        Tensor KT = Ops.transpose(K, 2, 3); // {B, H, dH, curT}
        Tensor scores = Ops.matmul(Q, KT);
        float scale = 1.0f / (float) Math.sqrt(headDim);
        scores = Ops.scale(scores, scale);

        if (useCausal) {
            Tensor mask = Ops.causalMask(curT);
            if (pastLen > 0) {
                // For generation with cache: only mask the new positions
                // We need a mask that allows attending to all past + current
                mask = adjustCausalMask(mask, T, pastLen);
            }
            scores = Ops.add(scores, mask);
        }

        // Apply key padding mask if provided
        if (keyPaddingMask != null) {
            // keyPaddingMask: {B, T} where true = mask (padding)
            // Expand to {B, 1, 1, curT} for broadcasting with scores {B, H, T, curT}
            scores = applyKeyPaddingMask(scores, keyPaddingMask, T, curT);
        }

        Tensor attn = Ops.softmax(scores); // {B, H, T, curT}
        attn = attnDropout.forward(attn, true); // attention dropout
        Tensor outHeads = Ops.matmul(attn, V); // {B, H, T, dH}
        Tensor out = unreshapeHeads(outHeads, B, T, dModel);
        Tensor projected = outProj.forward(out);
        projected = projDropout.forward(projected, true); // projection dropout

        return new Tensor[]{projected, K, V};
    }

    private Tensor applyRoPE(Tensor x, int offset) {
        // x: {B, H, T, dH}
        int B = x.size(0), H = x.size(1), T = x.size(2), dH = x.size(3);
        Tensor out = Tensor.zeros(B, H, T, dH);
        float[] xData = x.toFlat();
        float[] cosData = ropeCos.toFlat();
        float[] sinData = ropeSin.toFlat();
        float[] outData = out.toFlat();

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                for (int t = 0; t < T; t++) {
                    int pos = offset + t;
                    if (pos >= ropeCos.size(0)) {
                        // Extrapolate if beyond precomputed
                        pos = ropeCos.size(0) - 1;
                    }
                    for (int i = 0; i < dH; i += 2) {
                        int idx = ((b * H + h) * T + t) * dH + i;
                        float x1 = xData[idx];
                        float x2 = xData[idx + 1];
                        int posIdx = pos * dH + i;
                        float c = cosData[posIdx];
                        float s = sinData[posIdx];
                        outData[idx] = x1 * c - x2 * s;
                        outData[idx + 1] = x1 * s + x2 * c;
                    }
                }
            }
        }
        return out;
    }

    private Tensor concatCache(Tensor past, Tensor curr) {
        // past: {B, H, pastT, dH}, curr: {B, H, T, dH}
        // We need to concatenate along dimension 2
        // For simplicity, we'll use a new tensor and copy
        int B = past.size(0), H = past.size(1);
        int pastT = past.size(2), currT = curr.size(2), dH = past.size(3);
        Tensor out = Tensor.zeros(B, H, pastT + currT, dH);
        float[] pastData = past.toFlat();
        float[] currData = curr.toFlat();
        float[] outData = out.toFlat();

        int pastStride = H * pastT * dH;
        int currStride = H * currT * dH;
        int outStride = H * (pastT + currT) * dH;

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                int pastBase = (b * H + h) * pastT * dH;
                int currBase = (b * H + h) * currT * dH;
                int outBase = (b * H + h) * (pastT + currT) * dH;

                System.arraycopy(pastData, pastBase, outData, outBase, pastT * dH);
                System.arraycopy(currData, currBase, outData, outBase + pastT * dH, currT * dH);
            }
        }
        return out;
    }

    private Tensor adjustCausalMask(Tensor mask, int T, int pastLen) {
        // mask is {curT, curT} from causalMask(curT)
        // We need a mask of shape {T, curT} where T is query positions (1 for generation)
        // For generation with cache: the new token can attend to all past + itself
        // Create a mask of shape {T, curT} with all zeros (allow all)
        int curT = mask.size(0);
        float[] data = new float[T * curT];
        // All zeros = allow all positions
        return Tensor.of(data, T, curT);
    }

    private Tensor applyKeyPaddingMask(Tensor scores, boolean[][] keyPaddingMask, int T, int curT) {
        // scores: {B, H, T, curT}
        // keyPaddingMask: {B, T} where true = mask (padding)
        // We need to expand to {B, 1, 1, curT} and add -1e9 where mask is true
        int B = scores.size(0);
        int H = scores.size(1);
        float[] scoresData = scores.toFlat();
        float negInf = -1e9f;

        for (int b = 0; b < B; b++) {
            for (int t = 0; b < keyPaddingMask.length && t < keyPaddingMask[b].length; t++) {
                if (keyPaddingMask[b][t]) {
                    // Mask all attention from this query position
                    int base = ((b * H) * T + t) * curT;
                    for (int h = 0; h < H; h++) {
                        int idx = base + h * T * curT;
                        for (int j = 0; j < curT; j++) {
                            scoresData[idx + j] = negInf;
                        }
                    }
                }
            }
        }
        return scores;
    }

    /**
     * Convenience for encoder (self-attention, no causal mask).
     */
    public Tensor forward(Tensor x) {
        return forward(x, false);
    }

    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(qProj.parameters());
        p.addAll(kProj.parameters());
        p.addAll(vProj.parameters());
        p.addAll(outProj.parameters());
        return p;
    }

    private Tensor reshapeHeads(Tensor x, int B, int T) {
        Tensor flat = Ops.reshape(x, B, T, numHeads, headDim);
        return Ops.transpose(flat, 1, 2); // {B, H, T, dH}
    }

    private Tensor unreshapeHeads(Tensor x, int B, int T,
                                  int dModel) {
        Tensor tr = Ops.transpose(x, 1, 2); // {B, T, H, dH}
        return Ops.reshape(tr, B, T, dModel);
    }
}
