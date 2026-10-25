package transformers;

import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Transformer encoder (BERT-style): stack of TransformerBlock
 * layers with no causal masking. Used for bidirectional
 * sequence encoding.
 */
public final class TransformerEncoder {

    private final Embedding embedding;
    private final PositionalEmbedding positionalEmbedding;
    private final List<TransformerBlock> blocks;
    private final LayerNorm finalNorm;
    private final int numLayers;
    private final int dModel;
    private final int vocabSize;
    private final int maxSeqLen;
    private final boolean useRoPE;

    public TransformerEncoder(int vocabSize, int dModel,
                              int numHeads, int numLayers,
                              int ffnHidden, long seed) {
        this(vocabSize, dModel, numHeads, numLayers, ffnHidden, seed, false, 512);
    }

    /**
     * Creates an encoder from a BertConfig.
     */
    public TransformerEncoder(BertConfig config, long seed) {
        this(config.vocabSize, config.hiddenSize, config.numAttentionHeads,
             config.numHiddenLayers, config.intermediateSize, seed,
             false, config.maxPositionEmbeddings);
    }

    /**
     * Creates an encoder from a base TransformerConfig (for serialization).
     */
    public TransformerEncoder(TransformerConfig config, long seed) {
        this(config.getVocabSize(), config.getHiddenSize(), config.getNumAttentionHeads(),
             config.getNumHiddenLayers(), config.getIntermediateSize(), seed,
             false, config.getMaxPositionEmbeddings());
    }

    public TransformerEncoder(int vocabSize, int dModel,
                              int numHeads, int numLayers,
                              int ffnHidden, long seed,
                              boolean useRoPE, int maxSeqLen) {
        if (vocabSize <= 0 || dModel <= 0 || numHeads <= 0
                || numLayers <= 0 || ffnHidden <= 0) {
            throw new IllegalArgumentException(
                "all parameters must be positive");
        }
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.maxSeqLen = maxSeqLen;
        this.useRoPE = useRoPE;

        this.embedding = new Embedding(vocabSize, dModel, seed + 100);
        this.positionalEmbedding = new PositionalEmbedding(vocabSize, dModel, maxSeqLen,
            2, false, seed + 200);
        this.blocks = new ArrayList<>(numLayers);
        for (int i = 0; i < numLayers; i++) {
            long s = seed + 200 + i * 10;
            blocks.add(new TransformerBlock(dModel, numHeads,
                ffnHidden, s, useRoPE, maxSeqLen));
        }
        this.finalNorm = new LayerNorm(dModel);
    }

    /**
     * Creates an encoder from a BertConfig with positional embeddings.
     */
    public static TransformerEncoder fromBertConfig(BertConfig config, long seed) {
        return new TransformerEncoder(config.vocabSize, config.hiddenSize,
            config.numAttentionHeads, config.numHiddenLayers,
            config.intermediateSize, seed, false, config.maxPositionEmbeddings);
    }

    /**
     * Encodes a batch of token sequences.
     *
     * @param ids       token ids of length B*T
     * @param batchSize batch size
     * @param seqLen    sequence length
     * @return encoder hidden states of shape {B, T, dModel}
     */
    public Tensor forward(int[] ids, int batchSize,
                          int seqLen) {
        return forward(ids, batchSize, seqLen, null);
    }

    public Tensor forward(int[] ids, int[] segmentIds,
                          int batchSize, int seqLen) {
        return forward(ids, segmentIds, batchSize, seqLen, null);
    }

    /**
     * Encodes a batch of token sequences with padding mask.
     *
     * @param ids               token ids of length B*T
     * @param batchSize         batch size
     * @param seqLen            sequence length
     * @param keyPaddingMask    boolean mask of shape {B, T} where true = mask (padding)
     * @return encoder hidden states of shape {B, T, dModel}
     */
    public Tensor forward(int[] ids, int batchSize,
                          int seqLen, boolean[][] keyPaddingMask) {
        Tensor x = embedding.forward(ids, batchSize, seqLen);
        x = positionalEmbedding.forward(x, null, batchSize, seqLen);
        for (int i = 0; i < numLayers; i++) {
            x = blocks.get(i).forward(x, false, keyPaddingMask);
        }
        return finalNorm.forward(x);
    }

    public Tensor forward(int[] ids, int[] segmentIds,
                          int batchSize, int seqLen,
                          boolean[][] keyPaddingMask) {
        Tensor x = embedding.forward(ids, batchSize, seqLen);
        x = positionalEmbedding.forward(ids, segmentIds, batchSize, seqLen);
        for (int i = 0; i < numLayers; i++) {
            x = blocks.get(i).forward(x, false, keyPaddingMask);
        }
        return finalNorm.forward(x);
    }

    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(embedding.parameters());
        p.addAll(positionalEmbedding.parameters());
        for (TransformerBlock b : blocks) {
            p.addAll(b.parameters());
        }
        p.addAll(finalNorm.parameters());
        return p;
    }
}
