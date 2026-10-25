package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Transformer decoder (GPT-style) with RoPE, LM head, and generation.
 */
public final class TransformerDecoder {

    private final Embedding embedding;
    private final List<TransformerBlock> blocks;
    private final LayerNorm finalNorm;
    private final Linear lmHead;
    private final int numLayers;
    private final int dModel;
    private final int vocabSize;
    private final int maxSeqLen;
    private final boolean useRoPE;

    public TransformerDecoder(int vocabSize, int dModel,
                              int numHeads, int numLayers,
                              int ffnHidden, long seed) {
        this(vocabSize, dModel, numHeads, numLayers, ffnHidden, seed, false, 2048);
    }

    /**
     * Creates a decoder from a GPTConfig.
     */
    public TransformerDecoder(GPTConfig config, long seed) {
        this(config.vocabSize, config.hiddenSize, config.numAttentionHeads,
             config.numHiddenLayers, config.intermediateSize, seed,
             true, config.maxPositionEmbeddings);
    }

    /**
     * Creates a decoder from a base TransformerConfig (for serialization).
     */
    public TransformerDecoder(TransformerConfig config, long seed) {
        this(config.getVocabSize(), config.getHiddenSize(), config.getNumAttentionHeads(),
             config.getNumHiddenLayers(), config.getIntermediateSize(), seed,
             true, config.getMaxPositionEmbeddings());
    }

    /**
     * Creates a decoder with optional RoPE and LM head.
     */
    public TransformerDecoder(int vocabSize, int dModel,
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
        this.blocks = new ArrayList<>(numLayers);
        for (int i = 0; i < numLayers; i++) {
            long s = seed + 200 + i * 10;
            blocks.add(new TransformerBlock(dModel, numHeads,
                ffnHidden, s, useRoPE, maxSeqLen));
        }
        this.finalNorm = new LayerNorm(dModel);
        this.lmHead = new Linear(dModel, vocabSize, seed + 1000);
    }

    /**
     * Forward pass with optional KV cache and padding mask.
     *
     * @param ids               token ids of length B*T
     * @param batchSize         batch size
     * @param seqLen            sequence length
     * @param pastK             list of cached keys per layer (size = numLayers), or null
     * @param pastV             list of cached values per layer (size = numLayers), or null
     * @param pastLen           cached sequence length
     * @param keyPaddingMask    boolean mask of shape {B, T} where true = mask (padding)
     * @return logits {B, T, V}; cache lists are updated in place if not null
     */
    public Tensor forward(int[] ids, int batchSize, int seqLen,
                          List<Tensor> pastK, List<Tensor> pastV, int pastLen,
                          boolean[][] keyPaddingMask) {
        Tensor x = embedding.forward(ids, batchSize, seqLen);

        for (int i = 0; i < numLayers; i++) {
            if (pastK != null) {
                Tensor k = pastK.get(i);
                Tensor v = pastV.get(i);
                Tensor[] blockOut = blocks.get(i).forward(x, true, k, v, pastLen, keyPaddingMask);
                x = blockOut[0];
                // Update cache in place
                pastK.set(i, blockOut[1]);
                pastV.set(i, blockOut[2]);
            } else {
                x = blocks.get(i).forward(x, true, keyPaddingMask);
            }
        }

        x = finalNorm.forward(x);
        return lmHead.forward(x); // {B, T, V}
    }

    /**
     * Forward pass with optional KV cache (no padding mask).
     */
    public Tensor forward(int[] ids, int batchSize, int seqLen,
                          List<Tensor> pastK, List<Tensor> pastV, int pastLen) {
        return forward(ids, batchSize, seqLen, pastK, pastV, pastLen, null);
    }

    /**
     * Standard forward without cache.
     */
    public Tensor forward(int[] ids, int batchSize, int seqLen) {
        return forward(ids, batchSize, seqLen, null, null, 0);
    }

    /**
     * Autoregressive generation with KV cache.
     *
     * @param promptIds       initial token ids
     * @param maxNewTokens    maximum tokens to generate
     * @param temperature     sampling temperature (0.0 = greedy)
     * @param topK            top-K filtering (0 = disabled)
     * @param seed            RNG seed for sampling
     * @return generated token ids (prompt + new tokens)
     */
    public int[] generate(int[] promptIds, int maxNewTokens,
                          float temperature, int topK, long seed) {
        int promptLen = promptIds.length;
        if (promptLen > maxSeqLen) {
            throw new IllegalArgumentException(
                "prompt length " + promptLen + " exceeds maxSeqLen " + maxSeqLen);
        }

        // Initialize empty KV cache for each layer
        List<Tensor> pastK = new ArrayList<>(numLayers);
        List<Tensor> pastV = new ArrayList<>(numLayers);
        for (int i = 0; i < numLayers; i++) {
            pastK.add(null);
            pastV.add(null);
        }

        int[] allIds = new int[promptIds.length + maxNewTokens];
        System.arraycopy(promptIds, 0, allIds, 0, promptIds.length);
        int totalLen = promptIds.length;

        Random rng = new Random(seed);

        // Process prompt in one forward pass to build cache
        forward(promptIds, 1, promptLen, pastK, pastV, 0);

        // Generate new tokens
        for (int step = 0; step < maxNewTokens; step++) {
            int lastId = allIds[totalLen - 1];
            int[] single = {lastId};
            Tensor logits = forward(single, 1, 1, pastK, pastV, totalLen);

            float[] logitsData = logits.toFlat();
            int nextId = sample(logitsData, temperature, topK, new Random());
            allIds[totalLen++] = nextId;

            if (nextId == 50256) break; // EOS
            if (totalLen >= maxSeqLen) break;
        }

        int[] result = new int[totalLen];
        System.arraycopy(allIds, 0, result, 0, totalLen);
        return result;
    }

    private int sample(float[] logits, float temperature, int topK, Random rng) {
        if (temperature <= 0.0f) {
            int best = 0;
            for (int i = 1; i < logits.length; i++) {
                if (logits[i] > logits[best]) best = i;
            }
            return best;
        }
        for (int i = 0; i < logits.length; i++) logits[i] /= temperature;
        if (topK > 0 && topK < logits.length) {
            Integer[] indices = new Integer[logits.length];
            for (int i = 0; i < logits.length; i++) indices[i] = i;
            java.util.Arrays.sort(indices, (a, b) -> Float.compare(logits[b], logits[a]));
            float threshold = logits[indices[topK - 1]];
            for (int i = 0; i < logits.length; i++) {
                if (logits[i] < threshold) logits[i] = Float.NEGATIVE_INFINITY;
            }
        }
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        float sum = 0;
        for (int i = 0; i < logits.length; i++) {
            logits[i] = (float) Math.exp(logits[i] - max);
            sum += logits[i];
        }
        for (int i = 0; i < logits.length; i++) logits[i] /= sum;
        float r = new Random().nextFloat() * sum;
        float cum = 0;
        for (int i = 0; i < logits.length; i++) {
            cum += logits[i];
            if (r <= cum) return i;
        }
        return logits.length - 1;
    }

    public int vocabSize() { return vocabSize; }
    public int dModel() { return dModel; }
    public int maxSeqLen() { return maxSeqLen; }

    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(embedding.parameters());
        for (TransformerBlock b : blocks) {
            p.addAll(b.parameters());
        }
        p.addAll(finalNorm.parameters());
        p.addAll(lmHead.parameters());
        return p;
    }
}