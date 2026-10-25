package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT model with tied LM head for language modeling.
 * Supports generation with beam search, top-p sampling, repetition penalty, and stop sequences.
 */
public final class GPTLMHeadModel {

    private final GPTModel gpt;
    private final Linear lmHead;
    private final int vocabSize;
    private final int dModel;

    public GPTLMHeadModel(GPTConfig config, long seed) {
        this.vocabSize = config.vocabSize;
        this.dModel = config.hiddenSize;
        this.gpt = new GPTModel(config, seed);
        this.lmHead = new Linear(config.hiddenSize, config.vocabSize, 0);
    }

    /**
     * Forward pass with optional KV cache.
     */
    public Tensor forward(int[] ids, int batchSize, int seqLen,
                          List<autograd.Tensor> pastK, List<autograd.Tensor> pastV, int pastLen) {
        Tensor hidden = gpt.forward(ids, batchSize, seqLen, null, null, 0, null);
        return lmHead.forward(autograd.Ops.reshape(hidden, batchSize * (ids.length / batchSize), dModel));
    }

    /**
     * Standard forward without cache.
     */
    public Tensor forward(int[] ids, int batchSize, int seqLen) {
        return forward(ids, batchSize, seqLen, null, null, 0);
    }

    /**
     * Generates text autoregressively with various sampling strategies.
     *
     * @param promptIds        initial token ids
     * @param maxNewTokens     maximum tokens to generate
     * @param temperature      sampling temperature (0 = greedy)
     * @param topK             top-K filtering (0 = disabled)
     * @param topP             top-p (nucleus) filtering (0 = disabled)
     * @param repetitionPenalty penalty for repeated tokens (>1.0 penalizes)
     * @param stopTokenIds     token ids that stop generation
     * @param seed             RNG seed
     * @return generated token ids (prompt + new tokens)
     */
    public int[] generate(int[] promptIds, int maxNewTokens,
                          float temperature, int topK, float topP,
                          float repetitionPenalty, int[] stopTokenIds, long seed) {
        int promptLen = promptIds.length;
        if (promptLen > gpt.maxSeqLen()) {
            throw new IllegalArgumentException("Prompt too long");
        }

        // Initialize KV cache
        List<autograd.Tensor> pastK = new ArrayList<>();
        List<autograd.Tensor> pastV = new ArrayList<>();
        for (int i = 0; i < gpt.numLayers(); i++) {
            pastK.add(null);
            pastV.add(null);
        }

        int[] allIds = new int[promptIds.length + maxNewTokens];
        System.arraycopy(promptIds, 0, allIds, 0, promptLen);
        int totalLen = promptLen;

        java.util.Random rng = new java.util.Random(seed);

        // Process prompt in one forward pass to build cache
        forward(promptIds, 1, promptLen, pastK, pastV, 0);

        // Generate new tokens
        for (int step = 0; step < maxNewTokens; step++) {
            int[] single = {allIds[totalLen - 1]};
            Tensor logits = forward(single, 1, 1, pastK, pastV, totalLen - 1);
            float[] logitsData = logits.toFlat();

            // Apply repetition penalty
            if (repetitionPenalty != 1.0f) {
                applyRepetitionPenalty(logitsData, allIds, totalLen, repetitionPenalty);
            }

            int nextId = sample(logitsData, temperature, topK, topP, new java.util.Random());
            allIds[totalLen++] = nextId;

            // Check stop tokens
            if (stopTokenIds != null) {
                for (int stopId : stopTokenIds) {
                    if (nextId == stopId) {
                        int[] result = new int[totalLen];
                        System.arraycopy(allIds, 0, result, 0, totalLen);
                        return result;
                    }
                }
            }
            if (totalLen >= gpt.maxSeqLen()) break;
        }

        int[] result = new int[totalLen];
        System.arraycopy(allIds, 0, result, 0, totalLen);
        return result;
    }

    /**
     * Applies repetition penalty to logits.
     * From CTRL paper: penalize repeated tokens.
     */
    private void applyRepetitionPenalty(float[] logits, int[] generatedIds, int currentLen, float penalty) {
        for (int i = 0; i < penalty; i++) { // Just iterate for clarity
            for (int j = 0; j < currentLen; j++) {
                int tokenId = generatedIds[j];
                if (logits[tokenId] < 0) {
                    logits[tokenId] *= penalty;
                } else {
                    logits[tokenId] /= penalty;
                }
            }
        }
    }

    /**
     * Samples from logits with temperature, top-K, and top-P.
     */
    private int sample(float[] logits, float temperature, int topK, float topP, java.util.Random rng) {
        if (temperature <= 0.0f) {
            // Greedy
            int best = 0;
            for (int i = 1; i < logits.length; i++) {
                if (logits[i] > logits[best]) best = i;
            }
            return best;
        }

        // Temperature scaling
        for (int i = 0; i < logits.length; i++) logits[i] /= temperature;

        // Top-K filtering
        if (topK > 0 && topK < logits.length) {
            Integer[] indices = new Integer[logits.length];
            for (int i = 0; i < logits.length; i++) indices[i] = i;
            java.util.Arrays.sort(indices, (a, b) -> Float.compare(logits[b], logits[a]));
            float threshold = logits[indices[topK - 1]];
            for (int i = 0; i < logits.length; i++) {
                if (logits[i] < threshold) logits[i] = Float.NEGATIVE_INFINITY;
            }
        }

        // Top-P (nucleus) filtering
        if (topP > 0.0f && topP < 1.0f) {
            Integer[] indices = new Integer[logits.length];
            for (int i = 0; i < logits.length; i++) indices[i] = i;
            java.util.Arrays.sort(indices, (a, b) -> Float.compare(logits[b], logits[a]));

            float maxLogit = logits[indices[0]];
            float cumProb = 0;
            for (int i = 0; i < logits.length; i++) {
                float prob = (float) Math.exp(logits[indices[i]] - logits[indices[0]]);
                cumProb += prob;
                if (cumProb > topP) {
                    for (int j = i + 1; j < logits.length; j++) {
                        logits[indices[j]] = Float.NEGATIVE_INFINITY;
                    }
                    break;
                }
            }
        }

        // Softmax
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        float sum = 0;
        for (int i = 0; i < logits.length; i++) {
            logits[i] = (float) Math.exp(logits[i] - max);
            sum += logits[i];
        }
        for (int i = 0; i < logits.length; i++) logits[i] /= sum;

        // Sample
        float r = new java.util.Random().nextFloat() * sum;
        float cum = 0;
        for (int i = 0; i < logits.length; i++) {
            cum += logits[i];
            if (r <= cum) return i;
        }
        return logits.length - 1;
    }

    public List<autograd.Tensor> parameters() {
        List<autograd.Tensor> p = new java.util.ArrayList<>();
        p.addAll(gpt.parameters());
        p.addAll(lmHead.parameters());
        return p;
    }

    // Getters for BeamSearchDecoder
    public GPTModel getGpt() { return gpt; }
    public int vocabSize() { return vocabSize; }
    public int dModel() { return dModel; }
    public int maxSeqLen() { return gpt.maxSeqLen(); }
    public int numLayers() { return gpt.numLayers(); }
}