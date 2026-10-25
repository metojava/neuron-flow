package evaluation.nlp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ROUGE score for text summarization evaluation.
 * Computes ROUGE-N (n-gram overlap) and ROUGE-L (longest common subsequence).
 */
public final class RougeScore {

    private RougeScore() {
    }

    /**
     * ROUGE-N: n-gram recall between candidate and reference.
     *
     * @param candidate tokenized candidate
     * @param reference tokenized reference
     * @param n         n-gram order
     * @return ROUGE-N score in [0, 1]
     */
    public static double rougeN(String[] candidate, String[] reference, int n) {
        Map<String, Integer> refCounts = ngramCounts(reference, n);
        Map<String, Integer> candCounts = ngramCounts(candidate, n);

        int match = 0;
        int total = 0;
        for (Map.Entry<String, Integer> e : refCounts.entrySet()) {
            int candCount = candCounts.getOrDefault(e.getKey(), 0);
            match += Math.min(e.getValue(), candCount);
            total += e.getValue();
        }
        return (total == 0) ? 0.0 : (double) match / total;
    }

    /**
     * ROUGE-L: F-measure based on longest common subsequence.
     *
     * @param candidate tokenized candidate
     * @param reference tokenized reference
     * @return ROUGE-L F1 score in [0, 1]
     */
    public static double rougeL(String[] candidate, String[] reference) {
        int lcs = lcsLength(candidate, reference);
        if (lcs == 0) return 0.0;
        double precision = (double) lcs / candidate.length;
        double recall = (double) lcs / reference.length;
        if (precision + recall == 0) return 0.0;
        return 2.0 * precision * recall / (precision + recall);
    }

    private static int lcsLength(String[] a, String[] b) {
        int m = a.length, n = b.length;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (a[i - 1].equals(b[j - 1])) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
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
