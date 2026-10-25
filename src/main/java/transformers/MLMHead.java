package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.List;
import java.util.Random;

/**
 * Masked Language Modeling (MLM) head for BERT-style pretraining.
 * Predicts masked tokens using a linear layer tied to token embeddings.
 */
public final class MLMHead {

    private final Linear transform;
    private final LayerNorm layerNorm;
    private final Linear decoder;
    private final Tensor bias;
    private final int vocabSize;
    private final int dModel;

    /**
     * Creates an MLM head with tied output embeddings.
     *
     * @param vocabSize vocabulary size
     * @param dModel    model dimension
     * @param seed      RNG seed
     */
    public MLMHead(int vocabSize, int dModel, long seed) {
        if (vocabSize <= 0 || dModel <= 0) {
            throw new IllegalArgumentException("dimensions must be positive");
        }
        this.vocabSize = vocabSize;
        this.dModel = dModel;

        // Dense + LayerNorm + GELU (like BERT)
        this.transform = new Linear(dModel, dModel, 0);
        this.layerNorm = new LayerNorm(dModel);
        this.decoder = new Linear(dModel, vocabSize, 0);
        this.bias = Tensor.paramZeros(vocabSize);
    }

    /**
     * Ties the decoder weight to the token embedding matrix (weight tying).
     * Call after model construction.
     *
     * @param tokenEmbedding token embedding weight matrix of shape {V, D}
     */
    public void tieWeights(Tensor tokenEmbedding) {
        this.decoder.tieWeights(tokenEmbedding);
    }

    /**
     * Forward pass for MLM prediction.
     *
     * @param x            hidden states of shape {B, T, D} (or {M, D} for masked positions only)
     * @param maskedIndices indices of masked positions (length M), or null for all positions
     * @return logits of shape {M, V} where M = number of masked positions
     */
    public Tensor forward(Tensor x, int[] maskedIndices) {
        Tensor h = x;
        if (maskedIndices != null) {
            // Gather only masked positions: {M, D}
            h = Ops.gatherRows(x, maskedIndices);
        }
        Tensor h2 = transform.forward(h);
        h2 = Ops.gelu(h2);
        h2 = layerNorm.forward(h2);
        Tensor logits = decoder.forward(h2); // {M, V} or {B, T, V}
        // Add bias
        logits = Ops.add(logits, bias);
        return logits;
    }

    /**
     * Computes masked LM loss (cross-entropy over masked positions).
     *
     * @param logits        logits from forward, shape {M, V}
     * @param targetIds     target token ids for masked positions, length M
     * @return scalar loss
     */
    public static Tensor loss(Tensor logits, int[] targetIds) {
        // logits: {M, V}, targets: {M}
        return Ops.crossEntropy(logits, targetIds);
    }

    public int vocabSize() { return 0; } // set properly if needed

    public List<Tensor> parameters() {
        List<Tensor> p = new java.util.ArrayList<>();
        p.addAll(transform.parameters());
        p.addAll(layerNorm.parameters());
        p.addAll(decoder.parameters());
        p.add(bias);
        return p;
    }
}