package transformers;

import autograd.Tensor;

import java.util.List;

/**
 * BERT model for Masked Language Modeling.
 */
public final class BertForMaskedLM {

    private final BertModel bert;
    private final MLMHead mlmHead;

    public BertForMaskedLM(BertConfig config, long seed) {
        this.bert = new BertModel(config, seed);
        this.mlmHead = new MLMHead(config.vocabSize, config.hiddenSize, seed + 1000);
    }

    /**
     * Forward pass for MLM.
     *
     * @param inputIds      token ids [B, T] (already masked)
     * @param segmentIds    segment ids (optional)
     * @param attentionMask padding mask (optional)
     * @param labels        target token ids for masked positions [B, T] with -100 for non-masked
     * @return output with logits and optionally loss
     */
    public MaskedLMOutput forward(int[] inputIds, int[] segmentIds,
                                   int[] attentionMask, int[] labels) {
        BertModel.BertOutput bertOut = bert.forward(inputIds, segmentIds, null);
        autograd.Tensor logits = mlmHead.forward(bertOut.lastHiddenState, null);

        MaskedLMOutput output = new MaskedLMOutput(logits);
        if (labels != null) {
            output.loss = MLMHead.loss(logits, labels);
        }
        return output;
    }

    public List<autograd.Tensor> parameters() {
        List<autograd.Tensor> p = new java.util.ArrayList<>();
        p.addAll(bert.parameters());
        p.addAll(mlmHead.parameters());
        return p;
    }

    public static final class MaskedLMOutput {
        public final autograd.Tensor logits;
        public autograd.Tensor loss;

        public MaskedLMOutput(autograd.Tensor logits) {
            this.logits = logits;
        }
    }
}