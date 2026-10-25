package transformers;

import autograd.Tensor;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Quick demo of the transformer pipeline:
 * Tokenizer -> RoPE embeddings -> Causal LM head -> generate()
 */
public final class TransformerDemo {

    // Simple test tokenizer for demo
    private static final class TestTokenizer {
        private final Map<Character, Integer> charToId = new HashMap<>();
        private final Map<Integer, Character> idToChar = new HashMap<>();

        TestTokenizer() {
            // ASCII printable chars
            for (int i = 32; i < 127; i++) {
                char c = (char) i;
                charToId.put(c, c - 32);
                idToChar.put(c - 32, c);
            }
            charToId.put('\n', 127 - 32);
        }

        int[] encode(String text) {
            int[] ids = new int[text.length()];
            for (int i = 0; i < text.length(); i++) {
                ids[i] = charToId.getOrDefault(text.charAt(i), 0);
            }
            return ids;
        }

        String decode(int[] ids) {
            StringBuilder sb = new StringBuilder();
            for (int id : ids) {
                Character c = idToChar.get(id);
                sb.append(c != null ? c : '?');
            }
            return sb.toString();
        }

        int vocabSize() { return 96; }
    }

    public static void main(String[] args) {
        System.out.println("=== Transformer Pipeline Demo ===\n");

        // 1. Tokenizer
        System.out.println("1. Testing Tokenizer...");
        TestTokenizer tok = new TestTokenizer();
        int[] promptIds = tok.encode("Hello");
        System.out.println("   Prompt: \"Hello\" -> tokens: " + java.util.Arrays.toString(promptIds));

        // 2. Build small GPT-style model with RoPE
        System.out.println("\n2. Building TransformerDecoder (tiny: 2 layers, 64 dim, 4 heads)...");
        int vocabSize = 96;
        int dModel = 64;
        int numHeads = 4;
        int numLayers = 2;
        int ffnHidden = 256;
        int maxSeqLen = 128;

        TransformerDecoder model = new TransformerDecoder(
            vocabSize, dModel, 4, 2, 256, 42L, true, 128
        );
        long paramCount = model.parameters().stream().mapToLong(Tensor::numel).sum();
        System.out.println("   Params: " + paramCount);

        // 3. Forward pass (prompt processing)
        System.out.println("\n3. Forward pass with KV cache...");
        int[] promptIds3 = new int[]{'H'-32, 'e'-32, 'l'-32, 'l'-32, 'o'-32}; // "Hello"
        Tensor logits = model.forward(promptIds, 1, promptIds.length, null, null, 0);
        System.out.println("   Logits shape: " + java.util.Arrays.toString(logits.shape()));
        float[] logitsData = logits.toFlat();
        System.out.print("   Logits[0, -1, :5] = ");
        for (int i = 0; i < 5; i++) System.out.printf("%.4f ", logitsData[i]);
        System.out.println();

        // 4. Generation
        System.out.println("\n4. Generating (max 10 tokens, temp=0.8, topK=40)...");
        int[] promptIds2 = new int[]{'H'-32, 'e'-32, 'l'-32, 'l'-32, 'o'-32};
        int[] generated = model.generate(promptIds2, 10, 0.8f, 40, 123L);
        System.out.println("   Generated ids: " + java.util.Arrays.toString(generated));
        System.out.println("   Decoded: \"" + decodeTest(generated) + "\"");

        // 5. Test RoPE attention directly
        System.out.println("\n5. Testing RoPE attention directly...");
        var attn = new MultiHeadAttention(64, 4, 123L, true, 128, 0.0f, 0.0f);
        Tensor x = Tensor.of(new float[1 * 5 * 64], 1, 5, 64);
        Tensor out = attn.forward(x, true);
        System.out.println("   RoPE attention out shape: " + java.util.Arrays.toString(out.shape()));

        System.out.println("\n=== Demo complete ===");
    }

    private static String decodeTest(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (id >= 0 && id < 95) sb.append((char)(id + 32));
            else sb.append('?');
        }
        return sb.toString();
    }
}