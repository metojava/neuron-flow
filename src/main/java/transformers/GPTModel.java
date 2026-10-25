package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-style model with learned positional embeddings (not RoPE).
 * Matches Hugging Face GPT2Model architecture.
 */
public final class GPTModel {

    private final Embedding tokenEmbedding;
    private final Embedding positionEmbedding;
    private final List<TransformerBlock> blocks;
    private final LayerNorm finalNorm;
    private final int numLayers;
    private final int dModel;
    private final int vocabSize;
    private final int maxSeqLen;

    public GPTModel(GPTConfig config, long seed) {
        this.vocabSize = config.vocabSize;
        this.dModel = config.hiddenSize;
        this.numLayers = config.numHiddenLayers;
        this.maxSeqLen = config.maxPositionEmbeddings;

        this.tokenEmbedding = new Embedding(config.vocabSize, config.hiddenSize, seed + 100);
        this.positionEmbedding = new Embedding(config.maxPositionEmbeddings, config.hiddenSize, seed + 101);
        this.blocks = new ArrayList<>(config.numHiddenLayers);
        for (int i = 0; i < config.numHiddenLayers; i++) {
            long s = seed + 200 + i * 10;
            blocks.add(new TransformerBlock(config.hiddenSize, config.numAttentionHeads,
                config.intermediateSize, s, false, config.maxPositionEmbeddings,
                config.residPdrop));
        }
        this.finalNorm = new LayerNorm(config.hiddenSize);
    }

    /**
     * Forward pass with optional KV cache.
     *
     * @param ids               token ids [B*T]
     * @param batchSize         batch size
     * @param seqLen            sequence length
     * @param pastK             cached keys per layer
     * @param pastV             cached values per layer
     * @param pastLen           cached sequence length
     * @param attentionMask     attention mask (optional)
     * @return hidden states [B, T, D]
     */
    public Tensor forward(int[] ids, int batchSize, int seqLen,
                          List<autograd.Tensor> pastK, List<autograd.Tensor> pastV, int pastLen,
                          int[][] attentionMask) {
        Tensor x = tokenEmbedding.forward(ids, batchSize, seqLen);
        Tensor posEmbs = positionEmbedding.forward(createPositions(batchSize, seqLen), batchSize, seqLen);
        x = autograd.Ops.add(x, posEmbs);

        for (int i = 0; i < numLayers; i++) {
            if (pastK != null) {
                autograd.Tensor k = pastK.get(i);
                autograd.Tensor v = pastV.get(i);
                autograd.Tensor[] blockOut = blocks.get(i).forward(x, true, k, v, seqLen, null);
                x = blockOut[0];
                // Update cache in place if needed
            } else {
                x = blocks.get(i).forward(x, true, null);
            }
        }

        return finalNorm.forward(x);
    }

    /**
     * Forward pass without cache.
     */
    public Tensor forward(int[] ids, int batchSize, int seqLen) {
        return forward(ids, batchSize, ids.length / batchSize, null, null, 0, null);
    }

    private int[] createPositions(int batchSize, int seqLen) {
        int[] pos = new int[batchSize * seqLen];
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < seqLen; i++) {
                pos[b * seqLen + i] = i;
            }
        }
        return pos;
    }

    public List<autograd.Tensor> parameters() {
        List<autograd.Tensor> p = new ArrayList<>();
        p.addAll(tokenEmbedding.parameters());
        p.addAll(positionEmbedding.parameters());
        for (TransformerBlock b : blocks) {
            p.addAll(b.parameters());
        }
        p.addAll(finalNorm.parameters());
        return p;
    }

    public int maxSeqLen() { return maxSeqLen; }
    public int numLayers() { return numLayers; }
    public int vocabSize() { return vocabSize; }
    public int dModel() { return dModel; }
}