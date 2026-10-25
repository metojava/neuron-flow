package transformers;

import autograd.Tensor;

import java.nio.file.*;

/**
 * Quick test for ModelSerializer save/load.
 */
public final class SerializerTest {

    public static void main(String[] args) throws Exception {
        // Test with TransformerConfig directly (simpler)
        TransformerConfig config = new TransformerConfig();
        config.vocabSize = 96;
        config.hiddenSize = 64;
        config.numHiddenLayers = 2;
        config.numAttentionHeads = 4;
        config.intermediateSize = 256;
        config.maxPositionEmbeddings = 128;

        TransformerDecoder model = new TransformerDecoder(config, 42L);
        Path dir = Paths.get("test_model");

        // Save
        ModelSerializer.save(model, config, dir);
        System.out.println("Model saved to " + dir);

        // Load
        TransformerDecoder loaded = (TransformerDecoder) ModelSerializer.load(TransformerDecoder.class, TransformerConfig.class, Paths.get("test_model"));
        System.out.println("Model loaded");

        // Test forward pass matches
        int[] ids = new int[]{1, 2, 3, 4, 5};
        System.out.println("Original model params: " + model.parameters().size());
        for (int i = 0; i < model.parameters().size(); i++) {
            Tensor p = model.parameters().get(i);
            System.out.println("  Param " + i + ": shape=" + java.util.Arrays.toString(p.shape()) + " numel=" + p.numel());
        }

        System.out.println("Loaded model params: " + loaded.parameters().size());
        for (int i = 0; i < loaded.parameters().size(); i++) {
            Tensor p = loaded.parameters().get(i);
            System.out.println("  Param " + i + ": shape=" + java.util.Arrays.toString(p.shape()) + " numel=" + p.numel());
        }

        Tensor out1 = model.forward(ids, 1, ids.length, null, null, 0);
        Tensor out2 = loaded.forward(ids, 1, ids.length, null, null, 0);

        float[] data1 = out1.toFlat();
        float[] data2 = out2.toFlat();
        boolean match = java.util.Arrays.equals(data1, data2);

        System.out.println("Save/load test: " + (match ? "PASSED" : "FAILED"));
        if (!match) {
            System.out.println("Original: " + java.util.Arrays.toString(data1));
            System.out.println("Loaded:   " + java.util.Arrays.toString(out2.toFlat()));
        }
    }
}