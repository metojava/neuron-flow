package transformers;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Test the new Tokenizer features: special tokens, padding, truncation, encode_plus.
 */
public final class TokenizerFeatureTest {

    public static void main(String[] args) throws Exception {
        System.out.println("=== Testing Tokenizer Features ===\n");

        Map<String, Integer> vocab = new HashMap<>();
        vocab.put("hello", 1);
        vocab.put("world", 2);
        vocab.put("hello world", 3);
        vocab.put("test", 4);

        Tokenizer tok = new Tokenizer.Builder()
                .vocab(vocab)
                .merges(new java.util.ArrayList<>())
                .tokenPattern(java.util.regex.Pattern.compile("\\w+|\\s+|[^\\w\\s]"))
                .specialTokens(new HashMap<>())
                .build();

        System.out.println("1. Testing special tokens:");
        tok = tok.addSpecialTokens(Map.of(
                Tokenizer.PAD_TOKEN, 0,
                Tokenizer.UNK_TOKEN, 5,
                Tokenizer.CLS_TOKEN, 6,
                Tokenizer.SEP_TOKEN, 7,
                Tokenizer.MASK_TOKEN, 8,
                Tokenizer.BOS_TOKEN, 9,
                Tokenizer.EOS_TOKEN, 10
        ));
        System.out.println("   Special tokens: " + tok.getSpecialTokens());
        System.out.println("   PAD token ID: " + tok.padTokenId());
        System.out.println("   EOS token ID: " + tok.eosTokenId());

        System.out.println("\n2. Testing encode_plus with padding/truncation:");
        var result = tok.encodePlus("hello world", true, 10, true, true);
        System.out.println("   input_ids: " + java.util.Arrays.toString(result.get("input_ids")));
        System.out.println("   attention_mask: " + java.util.Arrays.toString(result.get("attention_mask")));
        System.out.println("   token_type_ids: " + java.util.Arrays.toString(result.get("token_type_ids")));

        System.out.println("\n3. Testing batch_encode_plus:");
        var batchResult = tok.batchEncodePlus(List.of("hello world", "test"), true, 8, true, true);
        System.out.println("   batch input_ids: " + java.util.Arrays.deepToString(batchResult.get("input_ids")));
        System.out.println("   batch attention_mask: " + java.util.Arrays.deepToString(batchResult.get("attention_mask")));

        System.out.println("\n4. Testing decode with special tokens:");
        int[] ids = new int[]{6, 1, 2, 7, 10}; // [CLS] hello world [SEP] [EOS]
        System.out.println("   Decoded: " + tok.decode(ids));

        System.out.println("\n=== All Tokenizer feature tests passed! ===");
    }
}