package data.encoding;

import java.util.HashMap;
import java.util.Map;

/**
 * Label encoder: maps string labels to integer indices and back.
 * Fit on training labels; unknown labels map to -1.
 */
public final class LabelEncoder {

    private final Map<String, Integer> toIndex = new HashMap<>();
    private final java.util.List<String> toLabel = new java.util.ArrayList<>();

    /**
     * Fits the encoder on the given labels.
     *
     * @param labels array of string labels
     */
    public void fit(String[] labels) {
        toIndex.clear();
        toLabel.clear();
        for (String label : labels) {
            if (!toIndex.containsKey(label)) {
                toIndex.put(label, toLabel.size());
                toLabel.add(label);
            }
        }
    }

    /** Encodes a single label. Returns -1 if unknown. */
    public int encode(String label) {
        return toIndex.getOrDefault(label, -1);
    }

    /** Decodes an index back to its label. Returns null if out of range. */
    public String decode(int index) {
        if (index < 0 || index >= toLabel.size()) return null;
        return toLabel.get(index);
    }

    /** Number of distinct labels seen during fit. */
    public int numClasses() {
        return toLabel.size();
    }

    /** Encodes an array of labels. */
    public int[] encodeAll(String[] labels) {
        int[] result = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            result[i] = encode(labels[i]);
        }
        return result;
    }
}
