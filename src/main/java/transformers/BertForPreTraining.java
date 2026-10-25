package transformers;

import autograd.Tensor;

import java.util.List;

/**
 * BERT model for pretraining (MLM + NSP).
 */
public final class BertForPreTraining {

    private final BertModel bert;
    private final MLMHead mlmHead;
    private final NSPHead nspHead;

    public BertForPreTraining(BertConfig config, long seed) {
        this.bert = new BertModel(config, seed);
        this.mlmHead = new MLMHead(config.vocabSize, config.hiddenSize, seed + 1000);
        this.nspHead = new NSPHead(config.hiddenSize, seed + 1001);
    }

    /**
     * Forward pass for pretraining.
     *
     * @param inputIds      token ids [B, T] (already masked for MLM)
     * @param segmentIds    segment ids [B, T] (for NSP)
     * @param attentionMask padding mask (optional)
     * @param mlmLabels     MLM labels [B, T] with -100 for non-masked
     * @param nspLabels     NSP labels [B] (0=isNext, 1=notNext)
     * @return output with MLM logits, NSP logits, and optionally losses
     */
    public PreTrainingOutput forward(int[] inputIds, int[] segmentIds,
                                      int[] attentionMask,
                                      int[] mlmLabels, int[] nspLabels) {
        BertModel.BertOutput bertOut = bert.forward(inputIds, segmentIds, null);
        autograd.Tensor mlmLogits = mlmHead.forward(bertOut.lastHiddenState, null);
        autograd.Tensor nspLogits = nspHead.forward(bertOut.pooledOutput);

        PreTrainingOutput output = new PreTrainingOutput(mlmLogits, nspLogits);

        if (mlmLabels != null) {
            output.mlmLoss = MLMHead.loss(mlmLogits, mlmLabels);
        }
        if (nspLabels != null) {
            output.nspLoss = NSPHead.loss(nspLogits, nspLabels);
        }
        if (mlmLabels != null && nspLabels != null) {
            output.loss = autograd.Ops.add(output.mlmLoss, output.nspLoss);
        }

        return output;
    }

    public List<autograd.Tensor> parameters() {
        List<autograd.Tensor> p = new java.util.ArrayList<>();
        p.addAll(bert.parameters());
        p.addAll(mlmHead.parameters());
        p.addAll(nspHead.parameters());
        return p;
    }

    public static final class PreTrainingOutput {
        public final autograd.Tensor mlmLogits;
        public final autograd.Tensor nspLogits;
        public autograd.Tensor mlmLoss;
        public autograd.Tensor nspLoss;
        public autograd.Tensor loss;

        public PreTrainingOutput(autograd.Tensor mlmLogits, autograd.Tensor nspLogits) {
            this.mlmLogits = mlmLogits;
            this.nspLogits = nspLogits;
        }
    }
}