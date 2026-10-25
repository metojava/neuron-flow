package data.text;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Simple whitespace tokenizer with a vocabulary. Maps tokens to integer
 * indices and back. Unknown tokens map to index 0 (reserved for &lt;unk&gt;).
 */
public final class Tokenizer {

    private final Map<String, Integer> vocab = new HashMap<>();
    private final List<String> reverseVocab = new ArrayList<>();
    private final boolean lowercase;

    /**
     * @param lowercase if true, tokens are lowercased before lookup
     */
    public Tokenizer(boolean lowercase) {
        this.lowercase = lowercase;
        // Reserve index 0 for unknown tokens
        reverseVocab.add("<unk>");
        vocab.put("<unk>", 0);
    }

    /** Tokenizes with lowercasing enabled. */
    public Tokenizer() {
        this(true);
    }

    /**
     * Splits text on whitespace and punctuation.
     *
     * @param text input text
     * @return array of tokens
     */
    public String[] tokenize(String text) {
        if (text == null) return new String[0];
        String normalized = lowercase ? text.toLowerCase() : text;
        return normalized.split("[\\s\\p{Punct}]+");
    }

    /**
     * Builds a vocabulary from a corpus. Tokens appearing fewer than
     * {@code minCount} times are discarded.
     *
     * @param corpus   array of text documents
     * @param minCount minimum token frequency to include
     */
    public void buildVocab(String[] corpus, int minCount) {
        Map<String, Integer> counts = new HashMap<>();
        for (String doc : corpus) {
            for (String token : tokenize(doc)) {
                if (token.isEmpty()) continue;
                counts.merge(token, 1, Integer::sum);
            }
        }
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            if (e.getValue() >= minCount && !vocab.containsKey(e.getKey())) {
                vocab.put(e.getKey(), reverseVocab.size());
                reverseVocab.add(e.getKey());
            }
        }
    }

    /**
     * Encodes a text into a sequence of token indices.
     *
     * @param text input text
     * @return array of token indices
     */
    public int[] encode(String text) {
        String[] tokens = tokenize(text);
        int[] result = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            result[i] = vocab.getOrDefault(tokens[i], 0);
        }
        return result;
    }

    /**
     * Decodes a sequence of token indices back to text.
     *
     * @param indices token indices
     * @return space-joined token string
     */
    public String decode(int[] indices) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < indices.length; i++) {
            if (i > 0) sb.append(' ');
            int idx = indices[i];
            if (idx >= 0 && idx < reverseVocab.size()) {
                sb.append(reverseVocab.get(idx));
            } else {
                sb.append("<unk>");
            }
        }
        return sb.toString();
    }

    /** Returns the vocabulary size (including &lt;unk&gt;). */
    public int vocabSize() {
        return reverseVocab.size();
    }

    /** Returns the index of a token, or 0 if unknown. */
    public int indexOf(String token) {
        return vocab.getOrDefault(lowercase ? token.toLowerCase() : token, 0);
    }
}
