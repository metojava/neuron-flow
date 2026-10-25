package transformers;

import autograd.Tensor;

import java.util.List;

/**
 * BERT model for Next Sentence Prediction.
 */
public final class BertForNextSentencePrediction {

    private final BertModel bert;
    private final NSPHead nspHead;

    public BertForNextSentencePrediction(BertConfig config, long seed) {
        this.bert = new BertModel(config, seed);
        this.nspHead = new NSPHead(config.hiddenSize, seed + 1000);
    }

    /**
     * Forward pass.
     *
     * @param inputIds      token ids [B, T]
     * @param segmentIds    segment ids [B, T] (0=sentence A, 1=sentence B)
     * @param attentionMask padding mask (optional)
     * @param labels        NSP labels [B] (optional), 0=isNext, 1=notNext
     * @return output with logits and optionally loss
     */
    public NSPOutput forward(int[] inputIds, int[] segmentIds,
                              int[] attentionMask, int[] labels) {
        BertModel.BertOutput bertOut = bert.forward(inputIds, segmentIds, attentionMask);
        autograd.Tensor logits = nspHead.forward(bertOut.pooledOutput);

        NSPOutput output = new NSPOutput(logits);
        if (labels != null) {
            output.loss = NSPHead.loss(logits, labels);
        }
        return output;
    }

    public List<autograd.Tensor> parameters() {
        List<autograd.Tensor> p = new java.util.ArrayList<>();
        p.addAll(bert.parameters());
        p.addAll(nspHead.parameters());
        return p;
    }

    public static final class NSPOutput {
        public final autograd.Tensor logits;
        public autograd.Tensor loss;

        public NSPOutput(autograd.Tensor logits) {
            this.logits = logits;
        }
    }
}