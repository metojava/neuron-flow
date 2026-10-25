package transformers;

import autograd.Tensor;

import java.util.List;

/**
 * BERT model (encoder only) with pooling layer.
 * Matches Hugging Face BertModel architecture.
 */
public final class BertModel {

    private final TransformerEncoder encoder;
    private final PoolingLayer pooler;
    private final int hiddenSize;

    public BertModel(BertConfig config, long seed) {
        this(config.vocabSize, config.hiddenSize, config.numAttentionHeads,
             config.numHiddenLayers, config.intermediateSize, seed);
    }

    public BertModel(int vocabSize, int hiddenSize, int numAttentionHeads,
                     int numHiddenLayers, int intermediateSize, long seed) {
        this.hiddenSize = hiddenSize;
        this.encoder = new TransformerEncoder(vocabSize, hiddenSize,
            numAttentionHeads, numHiddenLayers, intermediateSize, seed);
        this.pooler = new PoolingLayer(hiddenSize);
    }

    /**
     * Forward pass.
     *
     * @param inputIds       token ids of shape [B, T]
     * @param segmentIds     segment ids (optional)
     * @param attentionMask  padding mask (optional)
     * @return last hidden states [B, T, D] and pooled output [B, D]
     */
    public BertOutput forward(int[] inputIds, int[] segmentIds,
                               int[] attentionMask) {
        int batchSize = 1;
        int seqLen = inputIds.length;
        if (segmentIds != null) {
            batchSize = segmentIds.length / inputIds.length;
        }

        Tensor lastHiddenState = encoder.forward(inputIds, segmentIds != null ? segmentIds : new int[inputIds.length],
            batchSize, inputIds.length / batchSize);

        // Pooler: take [CLS] token (first token)
        Tensor pooledOutput = pooler.forward(lastHiddenState);

        return new BertOutput(lastHiddenState, pooledOutput);
    }

    public List<autograd.Tensor> parameters() {
        return encoder.parameters();
    }

    public int hiddenSize() { return hiddenSize; }

    /**
     * Output of BertModel.
     */
    public static final class BertOutput {
        public final autograd.Tensor lastHiddenState; // [B, T, D]
        public final autograd.Tensor pooledOutput;    // [B, D]

        public BertOutput(autograd.Tensor lastHiddenState, autograd.Tensor pooledOutput) {
            this.lastHiddenState = lastHiddenState;
            this.pooledOutput = pooledOutput;
        }
    }
}