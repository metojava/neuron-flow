package evaluation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * AUC-ROC and precision-recall curve utilities.
 */
public final class AucMetrics {

    private AucMetrics() {
    }

    /**
     * Computes the area under the ROC curve for binary classification.
     *
     * @param scores  predicted scores (higher = more likely positive)
     * @param labels  true labels (1 = positive, 0 = negative)
     * @return AUC in [0, 1]
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double aucRoc(float[] scores, int[] labels) {
        if (scores == null || labels == null || scores.length != labels.length
            || scores.length == 0) {
            throw new IllegalArgumentException("Invalid scores or labels");
        }

        // Sort by score descending
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < scores.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Float.compare(scores[b], scores[a]));

        int nPos = 0, nNeg = 0;
        for (int label : labels) {
            if (label == 1) nPos++;
            else nNeg++;
        }
        if (nPos == 0 || nNeg == 0) {
            return 0.5; // undefined, return chance
        }

        // Mann-Whitney U statistic
        double rankSum = 0.0;
        int i = 0;
        while (i < order.length) {
            int j = i;
            while (j + 1 < order.length && scores[order[j + 1]] == scores[order[i]]) {
                j++;
            }
            // Average rank for tied scores
            double avgRank = (i + 1 + j + 1) / 2.0;
            for (int k = i; k <= j; k++) {
                if (labels[order[k]] == 1) {
                    rankSum += avgRank;
                }
            }
            i = j + 1;
        }

        return (rankSum - (double) nPos * (nPos + 1) / 2.0) / ((double) nPos * nNeg);
    }

    /**
     * Computes precision-recall curve points.
     *
     * @param scores  predicted scores (higher = more likely positive)
     * @param labels  true labels (1 = positive, 0 = negative)
     * @return array of {threshold, precision, recall} rows
     */
    public static double[][] precisionRecallCurve(float[] scores, int[] labels) {
        if (scores == null || labels == null || scores.length != labels.length) {
            throw new IllegalArgumentException("Invalid scores or labels");
        }

        int n = scores.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Float.compare(scores[b], scores[a]));

        int tp = 0, fp = 0;
        int totalPos = 0;
        for (int label : labels) if (label == 1) totalPos++;

        List<double[]> points = new ArrayList<>();
        points.add(new double[] { Double.MAX_VALUE, 1.0, (totalPos > 0 ? 1.0 : 0.0) });

        for (int i = 0; i < n; i++) {
            if (labels[order[i]] == 1) tp++;
            else fp++;
            double precision = (tp + fp > 0) ? (double) tp / (tp + fp) : 1.0;
            double recall = (totalPos > 0) ? (double) tp / totalPos : 0.0;
            points.add(new double[] { scores[order[i]], precision, recall });
        }

        return points.toArray(new double[0][]);
    }

}
