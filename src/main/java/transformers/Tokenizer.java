package transformers;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Byte-Pair Encoding (BPE) tokenizer compatible with tiktoken format.
 * Supports special tokens, padding, truncation, and batch encoding.
 *
 * <p>Usage:
 * {@code
 * Tokenizer tok = Tokenizer.fromTiktoken("vocab.json", "merges.txt");
 * tok.addSpecialTokens(Map.of("pad_token", "[PAD]"));
 * var result = tok.encodePlus("Hello world", true, 128);
 * int[] inputIds = result.get("input_ids");
 * int[] attentionMask = result.get("attention_mask");
 * String text = tok.decode(inputIds);
 * }
 */
public final class Tokenizer {

    private final Map<String, Integer> vocab;
    private final Map<Integer, String> idToToken;
    private final List<Map.Entry<String, String>> merges;
    private final Map<String, Integer> mergeRanks;
    private final Pattern tokenPattern;
    private final Map<String, Integer> specialTokens;
    private final Map<Integer, String> idToSpecialToken;

    /**
     * Standard special token names.
     */
    public static final String CLS_TOKEN = "[CLS]";
    public static final String SEP_TOKEN = "[SEP]";
    public static final String MASK_TOKEN = "[MASK]";
    public static final String PAD_TOKEN = "[PAD]";
    public static final String UNK_TOKEN = "[UNK]";
    public static final String EOS_TOKEN = "</s>";
    public static final String BOS_TOKEN = "<s>";

    private Tokenizer(Builder builder) {
        this.vocab = Map.copyOf(builder.vocab);
        this.idToToken = new HashMap<>();
        for (Map.Entry<String, Integer> e : builder.vocab.entrySet()) {
            this.idToToken.put(e.getValue(), e.getKey());
        }
        this.merges = List.copyOf(builder.merges);
        this.mergeRanks = new HashMap<>();
        for (int i = 0; i < builder.merges.size(); i++) {
            Map.Entry<String, String> m = builder.merges.get(i);
            this.mergeRanks.put(m.getKey() + " " + m.getValue(), i);
        }
        this.tokenPattern = builder.tokenPattern;
        this.specialTokens = Map.copyOf(builder.specialTokens);
        this.idToSpecialToken = new HashMap<>();
        for (Map.Entry<String, Integer> e : this.specialTokens.entrySet()) {
            this.idToSpecialToken.put(e.getValue(), e.getKey());
        }
    }

    /**
     * Builder for creating a Tokenizer with custom special tokens.
     */
    public static final class Builder {
        private Map<String, Integer> vocab = new HashMap<>();
        private List<Map.Entry<String, String>> merges = new ArrayList<>();
        private Pattern tokenPattern = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?<!\\S)|\\s+");
        private Map<String, Integer> specialTokens = new HashMap<>();

        public Builder vocab(Map<String, Integer> vocab) {
            this.vocab = new HashMap<>(vocab);
            return this;
        }

        public Builder merges(List<Map.Entry<String, String>> merges) {
            this.merges = new ArrayList<>(merges);
            return this;
        }

        public Builder tokenPattern(Pattern tokenPattern) {
            this.tokenPattern = tokenPattern;
            return this;
        }

        public Builder specialTokens(Map<String, Integer> specialTokens) {
            this.specialTokens = new HashMap<>(specialTokens);
            return this;
        }

        public Builder addSpecialToken(String token, int id) {
            this.specialTokens.put(token, id);
            return this;
        }

        public Tokenizer build() {
            // Ensure special tokens are in vocab
            for (Map.Entry<String, Integer> e : specialTokens.entrySet()) {
                if (!vocab.containsKey(e.getKey())) {
                    vocab.put(e.getKey(), e.getValue());
                }
            }
            return new Tokenizer(this);
        }
    }

    /**
     * Creates a tokenizer from tiktoken vocab.json and merges.txt files.
     * Files must be on the classpath (e.g., resources/tokenizer/).
     */
    public static Tokenizer fromTiktoken(String vocabPath,
                                         String mergesPath) throws IOException {
        Map<String, Integer> vocab = loadVocab(vocabPath);
        List<Map.Entry<String, String>> merges = loadMerges(mergesPath);
        // GPT-2 / tiktoken regex pattern
        Pattern pat = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?<!\\S)|\\s+");
        return new Builder()
                .vocab(vocab)
                .merges(merges)
                .tokenPattern(pat)
                .build();
    }

    /**
     * Adds special tokens to the tokenizer.
     * If the token already exists in vocab, its ID is preserved.
     * Otherwise, a new ID is assigned (next available after vocab size).
     *
     * @param specialTokens map of token name to desired ID (or -1 for auto-assign)
     * @return this tokenizer for chaining
     */
    public Tokenizer addSpecialTokens(Map<String, Integer> specialTokens) {
        Map<String, Integer> newSpecial = new HashMap<>(this.specialTokens);
        int nextId = vocab.size();
        for (Map.Entry<String, Integer> e : specialTokens.entrySet()) {
            String token = e.getKey();
            int desiredId = e.getValue();
            if (vocab.containsKey(token)) {
                newSpecial.put(token, vocab.get(token));
            } else {
                int id = (desiredId >= 0) ? desiredId : nextId++;
                if (vocab.containsValue(id)) {
                    // find next available
                    while (vocab.containsValue(nextId)) nextId++;
                    id = nextId++;
                }
                // Note: vocab is immutable, would need to rebuild for true dynamic addition
                // For now, we track in specialTokens map
                newSpecial.put(token, id);
            }
        }
        // Return new tokenizer with updated special tokens
        return new Builder()
                .vocab(this.vocab)
                .merges(this.merges)
                .tokenPattern(this.tokenPattern)
                .specialTokens(newSpecial)
                .build();
    }

    /**
     * Returns a map of special token names to their IDs.
     */
    public Map<String, Integer> getSpecialTokens() {
        return Map.copyOf(specialTokens);
    }

    /**
     * Returns the ID of a special token, or -1 if not defined.
     */
    public int getSpecialTokenId(String name) {
        return specialTokens.getOrDefault(name, -1);
    }

    /**
     * Encodes text to token ids.
     */
    public int[] encode(String text) {
        List<Integer> ids = new ArrayList<>();
        var matcher = tokenPattern.matcher(text);
        while (matcher.find()) {
            String token = matcher.group();
            ids.addAll(bpeEncode(token));
        }
        return ids.stream().mapToInt(i -> i).toArray();
    }

    /**
     * Encodes text with padding/truncation and returns a map with:
     * - input_ids: token IDs
     * - attention_mask: 1 for real tokens, 0 for padding
     * - token_type_ids: segment IDs (0 for single sequence)
     *
     * @param text           input text
     * @param addSpecialTokens whether to add BOS/EOS tokens
     * @param maxLength      maximum sequence length (truncate/pad to this)
     * @param padding        true to pad to maxLength
     * @param truncation     true to truncate to maxLength
     * @return map with input_ids, attention_mask, token_type_ids
     */
    public Map<String, int[]> encodePlus(String text,
                                         boolean addSpecialTokens,
                                         int maxLength,
                                         boolean padding,
                                         boolean truncation) {
        int[] ids = encode(text);
        if (addSpecialTokens) {
            int bos = getSpecialTokenId(BOS_TOKEN);
            int eos = getSpecialTokenId(EOS_TOKEN);
            if (bos >= 0) ids = prepend(ids, bos);
            if (eos >= 0) ids = append(ids, eos);
        }
        if (truncation && ids.length > maxLength) {
            ids = Arrays.copyOf(ids, maxLength);
        }
        int[] attentionMask = new int[ids.length];
        Arrays.fill(attentionMask, 1);
        int[] tokenTypeIds = new int[ids.length]; // single sequence = all 0
        if (padding && ids.length < maxLength) {
            int pad = getSpecialTokenId(PAD_TOKEN);
            if (pad < 0) pad = 0;
            int padLen = maxLength - ids.length;
            ids = Arrays.copyOf(ids, maxLength);
            Arrays.fill(ids, ids.length - padLen, maxLength, pad);
            attentionMask = Arrays.copyOf(attentionMask, maxLength);
            Arrays.fill(attentionMask, attentionMask.length - padLen, maxLength, 0);
            tokenTypeIds = Arrays.copyOf(tokenTypeIds, maxLength);
        } else if (!padding && ids.length > maxLength) {
            ids = Arrays.copyOf(ids, maxLength);
            attentionMask = Arrays.copyOf(attentionMask, maxLength);
            tokenTypeIds = Arrays.copyOf(tokenTypeIds, maxLength);
        }
        return Map.of(
                "input_ids", ids,
                "attention_mask", attentionMask,
                "token_type_ids", tokenTypeIds
        );
    }

    /**
     * Batch encodes a list of texts with padding/truncation.
     * Returns a map with batched input_ids, attention_mask, token_type_ids.
     */
    public Map<String, int[][]> batchEncodePlus(List<String> texts,
                                                 boolean addSpecialTokens,
                                                 int maxLength,
                                                 boolean padding,
                                                 boolean truncation) {
        int batchSize = texts.size();
        int[][] allInputIds = new int[batchSize][];
        int[][] allAttentionMasks = new int[batchSize][];
        int[][] allTokenTypeIds = new int[batchSize][];

        for (int i = 0; i < batchSize; i++) {
            Map<String, int[]> result = encodePlus(texts.get(i), addSpecialTokens, maxLength, padding, truncation);
            allInputIds[i] = result.get("input_ids");
            allAttentionMasks[i] = result.get("attention_mask");
            allTokenTypeIds[i] = result.get("token_type_ids");
        }
        return Map.of(
                "input_ids", allInputIds,
                "attention_mask", allAttentionMasks,
                "token_type_ids", allTokenTypeIds
        );
    }

    /**
     * Decodes token ids back to text.
     */
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String token = idToToken.get(id);
            if (token != null) {
                sb.append(token);
            } else if (specialTokens.containsValue(id)) {
                // find special token name
                for (Map.Entry<String, Integer> e : specialTokens.entrySet()) {
                    if (e.getValue() == id) {
                        sb.append(e.getKey());
                        break;
                    }
                }
            }
        }
        // Convert byte tokens back
        String text = sb.toString();
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c >= 256 && c < 512) {
                out.append((char)(c - 256));
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }

    // --- Helpers ---

    private int[] prepend(int[] arr, int val) {
        int[] out = new int[arr.length + 1];
        out[0] = val;
        System.arraycopy(arr, 0, out, 1, arr.length);
        return out;
    }

    private int[] append(int[] arr, int val) {
        int[] out = Arrays.copyOf(arr, arr.length + 1);
        out[arr.length] = val;
        return out;
    }

    private List<Integer> bpeEncode(String token) {
        if (vocab.containsKey(token)) {
            return List.of(vocab.get(token));
        }
        List<String> word = new ArrayList<>();
        for (byte b : token.getBytes(StandardCharsets.UTF_8)) {
            word.add(byteToToken(b));
        }
        boolean merged;
        do {
            merged = false;
            int bestRank = Integer.MAX_VALUE;
            int bestIdx = -1;
            for (int i = 0; i < word.size() - 1; i++) {
                String pair = word.get(i) + " " + word.get(i + 1);
                Integer rank = mergeRanks.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }
            if (bestIdx != -1) {
                String mergedToken = word.get(bestIdx) + word.get(bestIdx + 1);
                word.set(bestIdx, mergedToken);
                word.remove(bestIdx + 1);
                merged = true;
            }
        } while (merged);

        List<Integer> result = new ArrayList<>();
        for (String w : word) {
            Integer id = vocab.get(w);
            if (id != null) {
                result.add(id);
            } else {
                for (byte b : w.getBytes(StandardCharsets.UTF_8)) {
                    String bt = byteToToken(b);
                    Integer bid = vocab.get(bt);
                    if (bid != null) result.add(bid);
                }
            }
        }
        return result;
    }

    private String byteToToken(byte b) {
        int v = b & 0xFF;
        if (v < 256) return new String(Character.toChars(256 + v));
        return "?";
    }

    // --- Loading helpers ---

    private static Map<String, Integer> loadVocab(String path) throws IOException {
        Map<String, Integer> vocab = new HashMap<>();
        try (InputStream is = Tokenizer.class.getClassLoader().getResourceAsStream(path)) {
            if (is == null) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(new java.io.FileInputStream(path), StandardCharsets.UTF_8))) {
                    parseJsonVocab(br, vocab);
                }
            } else {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(is, StandardCharsets.UTF_8))) {
                    parseJsonVocab(br, vocab);
                }
            }
        }
        return vocab;
    }

    private static void parseJsonVocab(BufferedReader br, Map<String, Integer> vocab) throws IOException {
        String line = br.readLine(); // skip {
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.equals("}")) continue;
            int colon = line.indexOf(':');
            if (colon == -1) continue;
            String token = line.substring(1, colon - 1).replace("\\u", "").replace("\\n", "\n");
            int id = Integer.parseInt(line.substring(colon + 1).replace(",", "").trim());
            token = token.replace("\\\"", "\"").replace("\\\\", "\\");
            vocab.put(token, id);
        }
    }

    private static List<Map.Entry<String, String>> loadMerges(String path) throws IOException {
        List<Map.Entry<String, String>> merges = new ArrayList<>();
        try (InputStream is = Tokenizer.class.getClassLoader().getResourceAsStream(path)) {
            if (is == null) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(new java.io.FileInputStream(path), StandardCharsets.UTF_8))) {
                    parseMerges(br, merges);
                }
            } else {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(is, StandardCharsets.UTF_8))) {
                    parseMerges(br, merges);
                }
            }
        }
        return merges;
    }

    private static void parseMerges(BufferedReader br, List<Map.Entry<String, String>> merges) throws IOException {
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] parts = line.split("\\s+");
            if (parts.length >= 2) {
                merges.add(Map.entry(parts[0], parts[1]));
            }
        }
    }

    public int vocabSize() { return vocab.size(); }
    public int eosTokenId() { return specialTokens.getOrDefault(EOS_TOKEN, -1); }
    public int padTokenId() { return specialTokens.getOrDefault(PAD_TOKEN, -1); }
}