package transformers;

/**
 * Configuration for BERT-style models.
 * Matches Hugging Face BertConfig defaults.
 */
public class BertConfig extends TransformerConfig {

    public BertConfig() {
        // BERT-base defaults
        this.vocabSize = 30522;
        this.hiddenSize = 768;
        this.numHiddenLayers = 12;
        this.numAttentionHeads = 12;
        this.intermediateSize = 3072;
        this.hiddenDropoutProb = 0.1f;
        this.attentionProbsDropoutProb = 0.1f;
        this.initializerRange = 0.02f;
        this.layerNormEps = 1e-12f;
        this.activationFunction = "gelu";
        this.maxPositionEmbeddings = 512;
        this.typeVocabSize = 2;
        this.padTokenId = 0;
        this.bosTokenId = 101;  // [CLS]
        this.eosTokenId = 102;  // [SEP]
        this.clsTokenId = 101;  // [CLS]
        this.sepTokenId = 102;  // [SEP]
        this.maskTokenId = 103; // [MASK]
        this.padTokenTypeId = 0;
    }

    /**
     * Creates a BERT config with custom values.
     */
    public static BertConfig of(int vocabSize, int hiddenSize, int numHiddenLayers,
                                 int numAttentionHeads, int intermediateSize) {
        BertConfig config = new BertConfig();
        config.vocabSize = vocabSize;
        config.hiddenSize = hiddenSize;
        config.numHiddenLayers = numHiddenLayers;
        config.numAttentionHeads = numAttentionHeads;
        config.intermediateSize = intermediateSize;
        return config;
    }

    /**
     * Predefined BERT configurations.
     */
    public static BertConfig bertBaseUncased() {
        return new BertConfig();
    }

    public static BertConfig bertLargeUncased() {
        BertConfig config = new BertConfig();
        config.hiddenSize = 1024;
        config.numHiddenLayers = 24;
        config.numAttentionHeads = 16;
        config.intermediateSize = 4096;
        return config;
    }

    public static BertConfig bertTiny() {
        BertConfig config = new BertConfig();
        config.hiddenSize = 128;
        config.numHiddenLayers = 2;
        config.numAttentionHeads = 2;
        config.intermediateSize = 512;
        return config;
    }

    public static BertConfig bertMini() {
        BertConfig config = new BertConfig();
        config.hiddenSize = 256;
        config.numHiddenLayers = 4;
        config.numAttentionHeads = 4;
        config.intermediateSize = 1024;
        return config;
    }

    public static BertConfig bertSmall() {
        BertConfig config = new BertConfig();
        config.hiddenSize = 512;
        config.numHiddenLayers = 4;
        config.numAttentionHeads = 8;
        config.intermediateSize = 2048;
        return config;
    }

    public static BertConfig bertMedium() {
        BertConfig config = new BertConfig();
        config.hiddenSize = 512;
        config.numHiddenLayers = 8;
        config.numAttentionHeads = 8;
        config.intermediateSize = 2048;
        return config;
    }
}