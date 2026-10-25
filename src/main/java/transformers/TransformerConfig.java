package transformers;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Base configuration class for transformer models.
 * Serializable for saving/loading.
 */
public class TransformerConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    protected int vocabSize;
    protected int hiddenSize;
    protected int numHiddenLayers;
    protected int numAttentionHeads;
    protected int intermediateSize;
    protected float hiddenDropoutProb;
    protected float attentionProbsDropoutProb;
    protected float initializerRange;
    protected float layerNormEps;
    protected String activationFunction;
    protected int maxPositionEmbeddings;
    protected int typeVocabSize;
    protected int padTokenId;
    protected int bosTokenId;
    protected int eosTokenId;
    protected int clsTokenId;
    protected int sepTokenId;
    protected int maskTokenId;
    protected int padTokenTypeId;

    public TransformerConfig() {
        // defaults
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
        this.bosTokenId = 2;
        this.eosTokenId = 3;
        this.clsTokenId = 101;
        this.sepTokenId = 102;
        this.maskTokenId = 103;
        this.padTokenTypeId = 0;
    }

    // Getters and setters
    public int getVocabSize() { return vocabSize; }
    public void setVocabSize(int vocabSize) { this.vocabSize = vocabSize; }

    public int getHiddenSize() { return hiddenSize; }
    public void setHiddenSize(int hiddenSize) { this.hiddenSize = hiddenSize; }

    public int getNumHiddenLayers() { return numHiddenLayers; }
    public void setNumHiddenLayers(int numHiddenLayers) { this.numHiddenLayers = numHiddenLayers; }

    public int getNumAttentionHeads() { return numAttentionHeads; }
    public void setNumAttentionHeads(int numAttentionHeads) { this.numAttentionHeads = numAttentionHeads; }

    public int getIntermediateSize() { return intermediateSize; }
    public void setIntermediateSize(int intermediateSize) { this.intermediateSize = intermediateSize; }

    public float getHiddenDropoutProb() { return hiddenDropoutProb; }
    public void setHiddenDropoutProb(float hiddenDropoutProb) { this.hiddenDropoutProb = hiddenDropoutProb; }

    public float getAttentionProbsDropoutProb() { return attentionProbsDropoutProb; }
    public void setAttentionProbsDropoutProb(float attentionProbsDropoutProb) { this.attentionProbsDropoutProb = attentionProbsDropoutProb; }

    public float getInitializerRange() { return initializerRange; }
    public void setInitializerRange(float initializerRange) { this.initializerRange = initializerRange; }

    public float getLayerNormEps() { return layerNormEps; }
    public void setLayerNormEps(float layerNormEps) { this.layerNormEps = layerNormEps; }

    public String getActivationFunction() { return activationFunction; }
    public void setActivationFunction(String activationFunction) { this.activationFunction = activationFunction; }

    public int getMaxPositionEmbeddings() { return maxPositionEmbeddings; }
    public void setMaxPositionEmbeddings(int maxPositionEmbeddings) { this.maxPositionEmbeddings = maxPositionEmbeddings; }

    public int getTypeVocabSize() { return typeVocabSize; }
    public void setTypeVocabSize(int typeVocabSize) { this.typeVocabSize = typeVocabSize; }

    public int getPadTokenId() { return padTokenId; }
    public void setPadTokenId(int padTokenId) { this.padTokenId = padTokenId; }

    public int getBosTokenId() { return bosTokenId; }
    public void setBosTokenId(int bosTokenId) { this.bosTokenId = bosTokenId; }

    public int getEosTokenId() { return eosTokenId; }
    public void setEosTokenId(int eosTokenId) { this.eosTokenId = eosTokenId; }

    public int getClsTokenId() { return clsTokenId; }
    public void setClsTokenId(int clsTokenId) { this.clsTokenId = clsTokenId; }

    public int getSepTokenId() { return sepTokenId; }
    public void setSepTokenId(int sepTokenId) { this.sepTokenId = sepTokenId; }

    public int getMaskTokenId() { return maskTokenId; }
    public void setMaskTokenId(int maskTokenId) { this.maskTokenId = maskTokenId; }

    public int getPadTokenTypeId() { return padTokenTypeId; }
    public void setPadTokenTypeId(int padTokenTypeId) { this.padTokenTypeId = padTokenTypeId; }

    /**
     * Converts config to a map for serialization.
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("vocab_size", vocabSize);
        map.put("hidden_size", hiddenSize);
        map.put("num_hidden_layers", numHiddenLayers);
        map.put("num_attention_heads", numAttentionHeads);
        map.put("intermediate_size", intermediateSize);
        map.put("hidden_dropout_prob", hiddenDropoutProb);
        map.put("attention_probs_dropout_prob", attentionProbsDropoutProb);
        map.put("initializer_range", initializerRange);
        map.put("layer_norm_eps", layerNormEps);
        map.put("activation_function", activationFunction);
        map.put("max_position_embeddings", maxPositionEmbeddings);
        map.put("type_vocab_size", typeVocabSize);
        map.put("pad_token_id", padTokenId);
        map.put("bos_token_id", bosTokenId);
        map.put("eos_token_id", eosTokenId);
        map.put("cls_token_id", clsTokenId);
        map.put("sep_token_id", sepTokenId);
        map.put("mask_token_id", maskTokenId);
        map.put("pad_token_type_id", padTokenTypeId);
        return map;
    }

    /**
     * Creates config from a map (for deserialization).
     */
    public static TransformerConfig fromMap(Map<String, Object> map) {
        TransformerConfig config = new TransformerConfig();
        if (map.containsKey("vocab_size")) config.vocabSize = ((Number) map.get("vocab_size")).intValue();
        if (map.containsKey("hidden_size")) config.hiddenSize = ((Number) map.get("hidden_size")).intValue();
        if (map.containsKey("num_hidden_layers")) config.numHiddenLayers = ((Number) map.get("num_hidden_layers")).intValue();
        if (map.containsKey("num_attention_heads")) config.numAttentionHeads = ((Number) map.get("num_attention_heads")).intValue();
        if (map.containsKey("intermediate_size")) config.intermediateSize = ((Number) map.get("intermediate_size")).intValue();
        if (map.containsKey("hidden_dropout_prob")) config.hiddenDropoutProb = ((Number) map.get("hidden_dropout_prob")).floatValue();
        if (map.containsKey("attention_probs_dropout_prob")) config.attentionProbsDropoutProb = ((Number) map.get("attention_probs_dropout_prob")).floatValue();
        if (map.containsKey("initializer_range")) config.initializerRange = ((Number) map.get("initializer_range")).floatValue();
        if (map.containsKey("layer_norm_eps")) config.layerNormEps = ((Number) map.get("layer_norm_eps")).floatValue();
        if (map.containsKey("activation_function")) config.activationFunction = (String) map.get("activation_function");
        if (map.containsKey("max_position_embeddings")) config.maxPositionEmbeddings = ((Number) map.get("max_position_embeddings")).intValue();
        if (map.containsKey("type_vocab_size")) config.typeVocabSize = ((Number) map.get("type_vocab_size")).intValue();
        if (map.containsKey("pad_token_id")) config.padTokenId = ((Number) map.get("pad_token_id")).intValue();
        if (map.containsKey("bos_token_id")) config.bosTokenId = ((Number) map.get("bos_token_id")).intValue();
        if (map.containsKey("eos_token_id")) config.eosTokenId = ((Number) map.get("eos_token_id")).intValue();
        if (map.containsKey("cls_token_id")) config.clsTokenId = ((Number) map.get("cls_token_id")).intValue();
        if (map.containsKey("sep_token_id")) config.sepTokenId = ((Number) map.get("sep_token_id")).intValue();
        if (map.containsKey("mask_token_id")) config.maskTokenId = ((Number) map.get("mask_token_id")).intValue();
        if (map.containsKey("pad_token_type_id")) config.padTokenTypeId = ((Number) map.get("pad_token_type_id")).intValue();
        return config;
    }

    @Override
    public String toString() {
        return "TransformerConfig{" +
                "vocabSize=" + vocabSize +
                ", hiddenSize=" + hiddenSize +
                ", numHiddenLayers=" + numHiddenLayers +
                ", numAttentionHeads=" + numAttentionHeads +
                ", intermediateSize=" + intermediateSize +
                '}';
    }
}