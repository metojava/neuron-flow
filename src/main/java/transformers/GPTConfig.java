package transformers;

/**
 * Configuration for GPT-2 style models.
 * Matches Hugging Face GPT2Config defaults.
 */
public class GPTConfig extends TransformerConfig {

    public GPTConfig() {
        // GPT-2 small defaults (124M params)
        this.vocabSize = 50257;
        this.hiddenSize = 768;           // n_embd
        this.numHiddenLayers = 12;        // n_layer
        this.numAttentionHeads = 12;      // n_head
        this.intermediateSize = 3072;     // n_inner (typically 4 * hidden_size)
        this.hiddenDropoutProb = 0.1f;    // resid_pdrop
        this.attentionProbsDropoutProb = 0.1f; // attn_pdrop
        this.initializerRange = 0.02f;
        this.layerNormEps = 1e-5f;        // layer_norm_epsilon
        this.activationFunction = "gelu_new"; // GPT-2 uses gelu_new
        this.maxPositionEmbeddings = 1024; // n_positions
        this.typeVocabSize = 0;           // No segment embeddings in GPT
        this.padTokenId = 50256;          // <|endoftext|>
        this.bosTokenId = 50256;          // <|endoftext|>
        this.eosTokenId = 50256;          // <|endoftext|>
        this.clsTokenId = -1;
        this.sepTokenId = -1;
        this.maskTokenId = -1;
        this.padTokenTypeId = 0;

        // GPT-2 specific
        this.nPositions = 1024;
        this.nEmbedd = 768;
        this.nLayer = 12;
        this.nHead = 12;
        this.nInner = 3072;
        this.residPdrop = 0.1f;
        this.embdPdrop = 0.1f;
        this.attnPdrop = 0.1f;
        this.layerNormEpsilon = 1e-5f;
        this.initializerRange = 0.02f;
        this.scaleAttnWeights = true;
        this.useCache = true;
    }

    // GPT-2 specific fields (aliases for HF compatibility)
    public int nPositions;
    public int nEmbedd;
    public int nLayer;
    public int nHead;
    public int nInner;
    public float residPdrop;
    public float embdPdrop;
    public float attnPdrop;
    public float layerNormEpsilon;
    public boolean scaleAttnWeights;
    public boolean useCache;

    /**
     * Creates a GPT config with custom values.
     */
    public static GPTConfig of(int vocabSize, int nEmbedd, int nLayer,
                                int nHead, int nInner, int nPositions) {
        GPTConfig config = new GPTConfig();
        config.vocabSize = vocabSize;
        config.hiddenSize = nEmbedd;
        config.nEmbedd = nEmbedd;
        config.numHiddenLayers = nLayer;
        config.nLayer = nLayer;
        config.numAttentionHeads = nHead;
        config.nHead = nHead;
        config.intermediateSize = nInner;
        config.nInner = nInner;
        config.maxPositionEmbeddings = nPositions;
        config.nPositions = nPositions;
        return config;
    }

    /**
     * Predefined GPT-2 configurations.
     */
    public static GPTConfig gpt2Small() { // 124M
        return new GPTConfig();
    }

    public static GPTConfig gpt2Medium() { // 350M
        GPTConfig config = new GPTConfig();
        config.hiddenSize = 1024;
        config.nEmbedd = 1024;
        config.numHiddenLayers = 24;
        config.nLayer = 24;
        config.numAttentionHeads = 16;
        config.nHead = 16;
        config.intermediateSize = 4096;
        config.nInner = 4096;
        return config;
    }

    public static GPTConfig gpt2Large() { // 774M
        GPTConfig config = new GPTConfig();
        config.hiddenSize = 1280;
        config.nEmbedd = 1280;
        config.numHiddenLayers = 36;
        config.nLayer = 36;
        config.numAttentionHeads = 20;
        config.nHead = 20;
        config.intermediateSize = 5120;
        config.nInner = 5120;
        return config;
    }

    public static GPTConfig gpt2XLarge() { // 1.5B
        GPTConfig config = new GPTConfig();
        config.hiddenSize = 1600;
        config.nEmbedd = 1600;
        config.numHiddenLayers = 48;
        config.nLayer = 48;
        config.numAttentionHeads = 25;
        config.nHead = 25;
        config.intermediateSize = 6400;
        config.nInner = 6400;
        return config;
    }

    @Override
    public String toString() {
        return "GPTConfig{" +
                "vocabSize=" + vocabSize +
                ", nEmbedd=" + nEmbedd +
                ", nLayer=" + nLayer +
                ", nHead=" + nHead +
                ", nInner=" + nInner +
                '}';
    }
}