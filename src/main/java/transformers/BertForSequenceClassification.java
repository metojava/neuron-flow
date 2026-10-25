package transformers;

import autograd.Tensor;

import java.util.List;

/**
 * BERT model for sequence classification.
 */
public final class BertForSequenceClassification {

    private final BertModel bert;
    private final Linear classifier;
    private final int numLabels;

    public BertForSequenceClassification(BertConfig config, int numLabels, long seed) {
        this.bert = new BertModel(config, seed);
        this.numLabels = numLabels;
        this.classifier = new Linear(config.hiddenSize, numLabels, 0);
    }

    /**
     * Forward pass.
     *
     * @param inputIds      token ids [B, T]
     * @param segmentIds    segment ids (optional)
     * @param attentionMask padding mask (optional)
     * @param labels        labels [B] (optional)
     * @return output with logits and optionally loss
     */
    public SequenceClassificationOutput forward(int[] inputIds,
                                                 int[] segmentIds,
                                                 int[] attentionMask,
                                                 int[] labels) {
        BertModel.BertOutput bertOut = bert.forward(inputIds, segmentIds, attentionMask);
        autograd.Tensor logits = classifier.forward(bertOut.pooledOutput);

        SequenceClassificationOutput output = new SequenceClassificationOutput(logits);
        if (labels != null) {
            output.loss = computeLoss(logits, labels);
        }
        return output;
    }

    private static autograd.Tensor computeLoss(autograd.Tensor logits, int[] labels) {
        return autograd.Ops.crossEntropy(logits, labels);
    }

    public List<autograd.Tensor> parameters() {
        List<autograd.Tensor> p = new java.util.ArrayList<>();
        p.addAll(bert.parameters());
        p.addAll(classifier.parameters());
        return p;
    }

    public static final class SequenceClassificationOutput {
        public final autograd.Tensor logits;
        public autograd.Tensor loss;

        public SequenceClassificationOutput(autograd.Tensor logits) {
            this.logits = logits;
        }
    }
}