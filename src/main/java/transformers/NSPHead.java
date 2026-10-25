package transformers;

import autograd.Tensor;

import java.util.List;

/**
 * Next Sentence Prediction (NSP) head for BERT.
 * Takes [CLS] pooled output and predicts isNext / notNext.
 */
public final class NSPHead {

    private final Linear classifier;

    public NSPHead(int hiddenSize, long seed) {
        this.classifier = new Linear(hiddenSize, 2, seed);
    }

    /**
     * @param pooledOutput [B, D] from BERT pooler
     * @return logits [B, 2] (isNext, notNext)
     */
    public Tensor forward(Tensor pooledOutput) {
        return classifier.forward(pooledOutput);
    }

    /**
     * Computes NSP loss (cross-entropy).
     *
     * @param logits  [B, 2]
     * @param labels  [B] with 0=isNext, 1=notNext
     * @return scalar loss
     */
    public static autograd.Tensor loss(autograd.Tensor logits, int[] labels) {
        return autograd.Ops.crossEntropy(logits, labels);
    }

    public List<autograd.Tensor> parameters() {
        return classifier.parameters();
    }
}