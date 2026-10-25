package transformers;

import autograd.Tensor;

import java.util.List;

/**
 * Perplexity evaluation for language models.
 * Computes average negative log-likelihood and perplexity.
 */
public final class PerplexityEvaluator {

    private final GPTLMHeadModel model;

    public PerplexityEvaluator(GPTLMHeadModel model) {
        this.model = model;
    }

    /**
     * Computes perplexity on a dataset.
     *
     * @param dataProvider provider yielding batches of (inputIds, labels, attentionMask)
     * @return PerplexityResult with average NLL, perplexity, and token count
     */
    public PerplexityResult evaluate(Trainer.DataProvider dataProvider) {
        double totalNLL = 0.0;
        long totalTokens = 0;
        int batchCount = 0;

        dataProvider.reset();
        while (dataProvider.hasNext()) {
            Trainer.Batch batch = dataProvider.next();

            // Forward pass
            Tensor logits = model.forward(batch.inputIds, 1, batch.inputIds.length,
                null, null, 0);

            // Compute cross-entropy loss
            // logits: [1, T, V], labels: [T] or [1, T]
            int seqLen = batch.inputIds.length;
            float[] logitsData = logits.toFlat();
            int vocabSize = model.vocabSize();

            // Reshape logits to [T, V]
            float[][] logits2D = new float[seqLen][];
            for (int t = 0; t < seqLen; t++) {
                logits2D[t] = new float[vocabSize];
                System.arraycopy(logitsData, t * vocabSize, logits2D[t], 0, vocabSize);
            }

            // Compute NLL for each position (ignoring padding if attentionMask provided)
            double batchNLL = 0.0;
            int validTokens = 0;

            for (int t = 0; t < seqLen; t++) {
                if (batch.attentionMask != null && t < batch.attentionMask.length && batch.attentionMask[t] == 0) {
                    continue; // Skip padding
                }
                int target = batch.labels[t];
                float[] logProbs = logSoftmax(logits2D[t]);
                batchNLL -= logProbs[target];
                validTokens++;
            }

            if (validTokens > 0) {
                totalNLL += batchNLL;
                totalTokens += validTokens;
                batchCount++;
            }
        }

        double avgNLL = totalTokens > 0 ? totalNLL / totalTokens : 0.0;
        double perplexity = Math.exp(avgNLL);

        return new PerplexityResult(avgNLL, perplexity, totalTokens, batchCount);
    }

    private float[] logSoftmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        float[] out = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            out[i] = (float) Math.exp(logits[i] - max);
        }
        float sum = 0;
        for (float v : out) sum += v;
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) Math.log(out[i] / sum);
        }
        return out;
    }

    public static final class PerplexityResult {
        public final double avgNLL;
        public final double perplexity;
        public final long totalTokens;
        public final int batchCount;

        public PerplexityResult(double avgNLL, double perplexity, long totalTokens, int batchCount) {
            this.avgNLL = avgNLL;
            this.perplexity = perplexity;
            this.totalTokens = totalTokens;
            this.batchCount = batchCount;
        }

        @Override
        public String toString() {
            return String.format("Perplexity: %.4f | Avg NLL: %.4f | Tokens: %d | Batches: %d",
                perplexity, avgNLL, totalTokens, batchCount);
        }
    }
}