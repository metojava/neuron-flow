package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * A single transformer block consisting of:
 * - Multi-head self-attention with pre-normalisation (LayerNorm)
 * - Add & norm
 * - Position-wise feed-forward
 * - Add & norm
 */
public final class TransformerBlock {

    private final LayerNorm attnNorm;
    private final MultiHeadAttention attention;
    private final LayerNorm ffnNorm;
    private final FeedForward ffn;
    private final Dropout residualDropout;
    private final int dModel;

public TransformerBlock(int dModel, int numHeads,
                        int ffnHidden, long seed) {
    this(dModel, numHeads, ffnHidden, seed, false, 2048, 0.1f);
}

public TransformerBlock(int dModel, int numHeads,
                        int ffnHidden, long seed,
                        boolean useRoPE, int maxSeqLen) {
    this(dModel, numHeads, ffnHidden, seed, useRoPE, maxSeqLen, 0.1f);
}

public TransformerBlock(int dModel, int numHeads,
                        int ffnHidden, long seed,
                        boolean useRoPE, int maxSeqLen,
                        float residualDropoutRate) {
    if (dModel <= 0 || numHeads <= 0 || ffnHidden <= 0) {
        throw new IllegalArgumentException(
            "all dimensions must be positive");
    }
    this.dModel = dModel;
    this.attnNorm = new LayerNorm(dModel);
    this.attention = new MultiHeadAttention(
        dModel, numHeads, seed, useRoPE, maxSeqLen, 0.1f, 0.1f);
    this.ffnNorm = new LayerNorm(dModel);
    this.ffn = new FeedForward(dModel, ffnHidden, seed + 5);
    this.residualDropout = new Dropout(residualDropoutRate, seed + 100);
}

    /**
     * Applies the transformer block.
     *
     * @param x                input of shape {B, T, D}
     * @param useCausal        apply causal mask (true for decoder)
     * @param keyPaddingMask   boolean mask of shape {B, T} where true = mask (padding)
     * @return output of shape {B, T, D}
     */
    public Tensor forward(Tensor x, boolean useCausal, boolean[][] keyPaddingMask) {
        return forward(x, useCausal, null, null, 0, keyPaddingMask)[0];
    }

    /**
     * Applies the transformer block.
     *
     * @param x         input of shape {B, T, D}
     * @param useCausal apply causal mask (true for decoder)
     * @return output of shape {B, T, D}
     */
    public Tensor forward(Tensor x, boolean useCausal) {
        return forward(x, useCausal, null);
    }

    /**
     * Forward with KV cache support.
     *
     * @param x                input {B, T, D}
     * @param useCausal        causal mask
     * @param pastK            cached K tensor (single, not list)
     * @param pastV            cached V tensor (single, not list)
     * @param pastLen          cached sequence length
     * @param keyPaddingMask   boolean mask of shape {B, T} where true = mask (padding)
     * @return [output, newK, newV]
     */
    public Tensor[] forward(Tensor x, boolean useCausal,
                            Tensor pastK, Tensor pastV, int pastLen,
                            boolean[][] keyPaddingMask) {
        // Pre-norm, residual with dropout
        Tensor h = attnNorm.forward(x);
        Tensor[] attnOut = attention.forward(h, useCausal, pastK, pastV, pastLen, keyPaddingMask);
        h = attnOut[0];
        h = residualDropout.forward(h, true);
        h = Ops.add(h, x);

        // Feed-forward
        Tensor h2 = ffnNorm.forward(h);
        h2 = ffn.forward(h2);
        h2 = residualDropout.forward(h2, true);
        Tensor out = Ops.add(h2, h);

        // Always return 3 elements when cache is expected (6-arg overload)
        return new Tensor[]{out, attnOut[1], attnOut[2]};
    }

    /**
     * Forward with KV cache support (no padding mask).
     *
     * @param x           input {B, T, D}
     * @param useCausal   causal mask
     * @param pastK       cached K tensor (single, not list)
     * @param pastV       cached V tensor (single, not list)
     * @param pastLen     cached sequence length
     * @return [output, newK, newV]
     */
    public Tensor[] forward(Tensor x, boolean useCausal,
                            Tensor pastK, Tensor pastV, int pastLen) {
        return forward(x, useCausal, pastK, pastV, pastLen, null);
    }

    public Tensor forward(Tensor x) {
        return forward(x, false);
    }

    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(attnNorm.parameters());
        p.addAll(attention.parameters());
        p.addAll(ffnNorm.parameters());
        p.addAll(ffn.parameters());
        return p;
    }
}
