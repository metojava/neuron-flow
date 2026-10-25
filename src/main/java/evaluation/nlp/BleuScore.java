package evaluation.nlp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * BLEU score for machine translation / text generation evaluation.
 * Computes n-gram precision with brevity penalty.
 */
public final class BleuScore {

    private BleuScore() {
    }

    /**
     * Computes BLEU score between a candidate and a reference.
     *
     * @param candidate tokenized candidate sentence
     * @param reference tokenized reference sentence
     * @param maxN      maximum n-gram order (typically 4)
     * @return BLEU score in [0, 1]
     */
    public static double bleu(String[] candidate, String[] reference, int maxN) {
        if (candidate.length == 0) return 0.0;

        double logSum = 0.0;
        int matches = 0;
        boolean anyMatch = false;

        for (int n = 1; n <= maxN; n++) {
            Map<String, Integer> refCounts = ngramCounts(reference, n);
            Map<String, Integer> candCounts = ngramCounts(candidate, n);

            int matchCount = 0;
            int totalCount = 0;
            for (Map.Entry<String, Integer> e : candCounts.entrySet()) {
                int refCount = refCounts.getOrDefault(e.getKey(), 0);
                matchCount += Math.min(e.getValue(), refCount);
                totalCount += e.getValue();
            }

            if (totalCount == 0) {
                return 0.0;
            }
            if (matchCount > 0) anyMatch = true;
            logSum += Math.log((double) matchCount / totalCount);
        }

        if (!anyMatch) return 0.0;

        // Brevity penalty
        double bp = 1.0;
        if (candidate.length < reference.length) {
            bp = Math.exp(1.0 - (double) reference.length / candidate.length);
        }

        return bp * Math.exp(logSum / maxN);
    }

    private static Map<String, Integer> ngramCounts(String[] tokens, int n) {
        Map<String, Integer> counts = new HashMap<>();
        for (int i = 0; i + n <= tokens.length; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (j > 0) sb.append(" ");
                sb.append(tokens[i + j]);
            }
            counts.merge(sb.toString(), 1, Integer::sum);
        }
        return counts;
    }
}
