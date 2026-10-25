package transformers;

import autograd.Tensor;

/**
 * Debug parameter count for TransformerDecoder.
 */
public final class DebugParams {

    public static void main(String[] args) {
        GPTConfig config = GPTConfig.gpt2Small();
        config.hiddenSize = 64;
        config.numHiddenLayers = 2;
        config.numAttentionHeads = 4;
        config.intermediateSize = 256;
        config.vocabSize = 96;
        config.maxPositionEmbeddings = 128;

        System.out.println("Config:");
        System.out.println("  vocabSize=" + config.vocabSize);
        System.out.println("  hiddenSize=" + config.hiddenSize);
        System.out.println("  numHiddenLayers=" + config.numHiddenLayers);
        System.out.println("  numAttentionHeads=" + config.numAttentionHeads);
        System.out.println("  intermediateSize=" + config.intermediateSize);
        System.out.println("  maxPositionEmbeddings=" + config.maxPositionEmbeddings);

        TransformerDecoder model = new TransformerDecoder(config, 42L);
        System.out.println("\nModel parameters: " + model.parameters().size());
        for (int i = 0; i < model.parameters().size(); i++) {
            Tensor p = model.parameters().get(i);
            System.out.println("  Param " + i + ": shape=" + java.util.Arrays.toString(p.shape()) + " numel=" + p.numel());
        }

        // Also test with TransformerConfig
        System.out.println("\n--- With TransformerConfig ---");
        TransformerConfig baseConfig = new TransformerConfig();
        baseConfig.vocabSize = 96;
        baseConfig.hiddenSize = 64;
        baseConfig.numHiddenLayers = 2;
        baseConfig.numAttentionHeads = 4;
        baseConfig.intermediateSize = 256;
        baseConfig.maxPositionEmbeddings = 128;

        TransformerDecoder model2 = new TransformerDecoder(baseConfig, 42L);
        System.out.println("Model2 parameters: " + model2.parameters().size());
        for (int i = 0; i < model2.parameters().size(); i++) {
            Tensor p = model2.parameters().get(i);
            System.out.println("  Param " + i + ": shape=" + java.util.Arrays.toString(p.shape()) + " numel=" + p.numel());
        }
    }
}