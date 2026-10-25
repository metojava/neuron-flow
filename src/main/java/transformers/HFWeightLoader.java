package transformers;

import autograd.Tensor;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Hugging Face weight loader.
 * Maps HF state_dict keys to our parameter names and loads weights.
 */
public final class HFWeightLoader {

    private HFWeightLoader() {}

    /**
     * Loads weights from a Hugging Face safetensors file into a model.
     *
     * @param model       model with parameters() method
     * @param safetensorsPath path to .safetensors file
     * @param modelType   "bert" or "gpt2" for key mapping
     * @throws IOException on I/O error
     */
    public static void loadWeights(Object model, Path safetensorsPath, String modelType) throws IOException {
        Map<String, autograd.Tensor> hfWeights = SafetensorsLoader.load(safetensorsPath);

        // Map HF keys to our parameter names
        Map<String, String> keyMap = getKeyMapping(modelType);
        
        // Get model parameters
        List<autograd.Tensor> params = getParameters(model);
        Map<String, autograd.Tensor> paramMap = new HashMap<>();
        for (Tensor p : params) {
            // Generate a name based on parameter shape/position
            // In practice, we'd need a more robust naming scheme
            paramMap.put(generateParamName(p), p);
        }

        // Load weights
        for (Map.Entry<String, String> mapping : keyMap.entrySet()) {
            String hfKey = mapping.getKey();
            String ourKey = mapping.getValue();

            autograd.Tensor hfTensor = hfWeights.get(hfKey);
            autograd.Tensor ourTensor = paramMap.get(ourKey);

            if (hfTensor != null && ourTensor != null) {
                if (ourTensor.numel() != hfTensor.numel()) {
                    System.err.printf("Shape mismatch for %s -> %s: expected %d, got %d%n",
                        hfKey, ourKey, ourTensor.numel(), hfTensor.numel());
                    continue;
                }
                // Copy data
                float[] src = hfTensor.toFlat();
                float[] dst = ourTensor.toFlat();
                System.arraycopy(src, 0, dst, 0, src.length);
            } else if (hfTensor == null) {
                System.err.println("HF weight not found: " + hfKey);
            } else {
                System.err.println("Model parameter not found: " + ourKey);
            }
        }
    }

    private static Map<String, String> getKeyMapping(String modelType) {
        Map<String, String> map = new HashMap<>();

        if ("bert".equalsIgnoreCase(modelType)) {
            // BERT key mappings
            // embeddings
            map.put("embeddings.word_embeddings.weight", "embedding.weight");
            map.put("embeddings.position_embeddings.weight", "positionalEmbedding.positionEmb.weight");
            map.put("embeddings.token_type_embeddings.weight", "positionalEmbedding.segmentEmb.weight");
            map.put("embeddings.LayerNorm.weight", "positionalEmbedding.finalNorm.gamma");
            map.put("embeddings.LayerNorm.bias", "positionalEmbedding.finalNorm.beta");

            // encoder layers
            for (int i = 0; i < 12; i++) { // BERT-base has 12 layers
                String l = "encoder.layer." + i + ".";
                String L = "encoder.blocks." + i + ".";

                // attention
                map.put(l + "attention.self.query.weight", L + "attention.qProj.weight");
                map.put(l + "attention.self.query.bias", L + "attention.qProj.bias");
                map.put(l + "attention.self.key.weight", L + "attention.kProj.weight");
                map.put(l + "attention.self.key.bias", L + "attention.kProj.bias");
                map.put(l + "attention.self.value.weight", L + "attention.vProj.weight");
                map.put(l + "attention.self.value.bias", L + "attention.vProj.bias");
                map.put(l + "attention.output.dense.weight", L + "attention.outProj.weight");
                map.put(l + "attention.output.dense.bias", L + "attention.outProj.bias");
                map.put(l + "attention.output.LayerNorm.weight", L + "attnNorm.gamma");
                map.put(l + "attention.output.LayerNorm.bias", L + "attnNorm.beta");

                // feed-forward
                map.put(l + "intermediate.dense.weight", L + "ffn.linear1.weight");
                map.put(l + "intermediate.dense.bias", L + "ffn.linear1.bias");
                map.put(l + "output.dense.weight", L + "ffn.linear2.weight");
                map.put(l + "output.dense.bias", L + "ffn.linear2.bias");
                map.put(l + "output.LayerNorm.weight", L + "ffnNorm.gamma");
                map.put(l + "output.LayerNorm.bias", L + "ffnNorm.beta");
            }

            // pooler
            map.put("pooler.dense.weight", "pooler.dense.weight");
            map.put("pooler.dense.bias", "pooler.dense.bias");
        } else if ("gpt2".equalsIgnoreCase(modelType)) {
            // GPT-2 key mappings
            map.put("wte.weight", "embedding.tokenEmbedding.weight");
            map.put("wpe.weight", "gpt.positionEmbedding.weight");

            for (int i = 0; i < 12; i++) { // GPT-2 small has 12 layers
                String l = "h." + i + ".";
                String L = "blocks." + i + ".";

                // attention
                map.put(l + "attn.c_attn.weight", L + "attention.qProj.weight"); // combined QKV
                map.put(l + "attn.c_attn.bias", L + "attention.qProj.bias");
                map.put(l + "attn.c_proj.weight", L + "attention.outProj.weight");
                map.put(l + "attn.c_proj.bias", L + "attention.outProj.bias");
                map.put(l + "ln_1.weight", L + "attnNorm.gamma");
                map.put(l + "ln_1.bias", L + "attnNorm.beta");

                // feed-forward
                map.put(l + "mlp.c_fc.weight", L + "ffn.linear1.weight");
                map.put(l + "mlp.c_fc.bias", L + "ffn.linear1.bias");
                map.put(l + "mlp.c_proj.weight", L + "ffn.linear2.weight");
                map.put(l + "mlp.c_proj.bias", L + "ffn.linear2.bias");
                map.put(l + "ln_2.weight", L + "ffnNorm.gamma");
                map.put(l + "ln_2.bias", L + "ffnNorm.beta");
            }

            // final
            map.put("ln_f.weight", "finalNorm.gamma");
            map.put("ln_f.bias", "finalNorm.beta");
        }

        return map;
    }

    private static List<autograd.Tensor> getParameters(Object model) {
        try {
            java.lang.reflect.Method m = model.getClass().getMethod("parameters");
            @SuppressWarnings("unchecked")
            List<autograd.Tensor> params = (List<autograd.Tensor>) m.invoke(model);
            return params;
        } catch (Exception e) {
            throw new IllegalStateException("Model must have parameters() method", e);
        }
    }

    private static String generateParamName(autograd.Tensor p) {
        // Simple naming based on shape - in practice use a proper naming scheme
        return "param_" + p.numel();
    }
}