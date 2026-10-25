package evaluation;

import java.io.PrintStream;

/**
 * Classification evaluation utilities: argmax decoding,
 * accuracy, confusion matrices and per-class precision /
 * recall / F1. Works on raw prediction matrices, so it can be
 * used with any network or any classifier.
 * Convention: confusion[actual][predicted].
 */
public final class ClassificationMetrics {

    private ClassificationMetrics() {
    }

    /**
     * Returns the index of the largest value.
     *
     * @param values scores, e.g. one per class
     * @return index of the maximum (ties: lowest index)
     * @throws IllegalArgumentException if null or empty
     */
    public static int argmax(float[] values) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException(
                "Output must not be empty");
        }

        int best = 0;

        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }

        return best;
    }

    /**
     * Accuracy from precomputed predictions.
     *
     * @param predicted predicted class per row, [N]
     * @param actual    true class per row, [N]
     * @return fraction correct in [0, 1]
     * @throws IllegalArgumentException on length mismatch
     */
    public static float accuracy(int[] predicted, int[] actual) {
        if (predicted == null || actual == null ||
            predicted.length != actual.length ||
            predicted.length == 0) {
            throw new IllegalArgumentException(
                "Prediction and label arrays must have equal nonzero length");
        }

        int correct = 0;

        for (int i = 0; i < predicted.length; i++) {
            if (predicted[i] == actual[i]) {
                correct++;
            }
        }

        return (float) correct / predicted.length;
    }

    /**
     * Builds a confusion matrix from predictions.
     *
     * @param predicted   predicted class per row, [N]
     * @param actual      true class per row, [N]
     * @param numClasses  number of classes (matrix size)
     * @return matrix where m[actual][predicted] is the count
     * @throws IllegalArgumentException on mismatch or out-of-range
     *                                  class ids
     */
    public static int[][] confusionMatrix(int[] predicted,
                                          int[] actual,
                                          int numClasses) {
        if (predicted == null || actual == null ||
            predicted.length != actual.length) {
            throw new IllegalArgumentException(
                "Prediction and label arrays must match");
        }

        int[][] matrix = new int[numClasses][numClasses];

        for (int i = 0; i < predicted.length; i++) {
            int p = predicted[i];
            int a = actual[i];

            if (p < 0 || p >= numClasses ||
                a < 0 || a >= numClasses) {
                throw new IllegalArgumentException(
                    "Class id out of range at row " + i);
            }

            matrix[a][p]++;
        }

        return matrix;
    }

    /**
     * Precision for one class: of all rows predicted as this
     * class, the fraction that truly belong to it.
     *
     * @param confusion confusion matrix, [actual][predicted]
     * @param c         class index
     * @return precision in [0, 1], 0 when nothing was predicted
     */
    public static float precision(int[][] confusion, int c) {
        int predictedCount = 0;

        for (int a = 0; a < confusion.length; a++) {
            predictedCount += confusion[a][c];
        }

        if (predictedCount == 0) {
            return 0.0f;
        }

        return (float) confusion[c][c] / predictedCount;
    }

    /**
     * Recall for one class: of all rows of this class, the
     * fraction that were predicted correctly.
     *
     * @param confusion confusion matrix, [actual][predicted]
     * @param c         class index
     * @return recall in [0, 1], 0 when the class never appears
     */
    public static float recall(int[][] confusion, int c) {
        int actualCount = 0;

        for (int p = 0; p < confusion.length; p++) {
            actualCount += confusion[c][p];
        }

        if (actualCount == 0) {
            return 0.0f;
        }

        return (float) confusion[c][c] / actualCount;
    }

    /**
     * Harmonic mean of precision and recall for one class.
     *
     * @param confusion confusion matrix, [actual][predicted]
     * @param c         class index
     * @return F1 in [0, 1]
     */
    public static float f1(int[][] confusion, int c) {
        float p = precision(confusion, c);
        float r = recall(confusion, c);

        if (p + r == 0.0f) {
            return 0.0f;
        }

        return 2.0f * p * r / (p + r);
    }

    /**
     * Prints a confusion matrix with an optional row-label
     * column and a summary of per-class precision / recall / F1.
     *
     * @param out        where to print (e.g. System.out)
     * @param confusion  matrix, [actual][predicted]
     * @param classNames one name per class; if null, indices are
     *                   used as labels
     */
    public static void printReport(PrintStream out,
                                   int[][] confusion,
                                   String[] classNames) {
        int n = confusion.length;

        out.print("           ");
        for (int c = 0; c < n; c++) {
            out.printf("%6d", c);
        }
        out.println();

        for (int actual = 0; actual < n; actual++) {
            String label = (classNames != null
                && actual < classNames.length)
                ? classNames[actual]
                : String.valueOf(actual);

            out.printf("%-12s ", label);

            for (int predicted = 0; predicted < n; predicted++) {
                out.printf("%6d", confusion[actual][predicted]);
            }

            out.println();
        }

        out.printf("%n%-12s %10s %10s %10s%n",
            "class", "precision", "recall", "f1");

        for (int c = 0; c < n; c++) {
            String label = (classNames != null
                && c < classNames.length)
                ? classNames[c]
                : String.valueOf(c);

            out.printf("%-12s %10.3f %10.3f %10.3f%n",
                label,
                precision(confusion, c),
                recall(confusion, c),
                f1(confusion, c));
        }
    }

    /**
     * F-beta score for class c. beta > 1 favours recall,
     * beta < 1 favours precision.
     */
    public static float fBeta(int[][] confusion, int c, float beta) {
        float p = precision(confusion, c);
        float r = recall(confusion, c);
        if (p + r == 0.0f) return 0.0f;
        float beta2 = beta * beta;
        return (1.0f + beta2) * p * r / (beta2 * p + r);
    }

    /** Macro-averaged F1 across all classes. */
    public static float macroF1(int[][] confusion) {
        int n = confusion.length;
        float sum = 0.0f;
        for (int c = 0; c < n; c++) sum += f1(confusion, c);
        return sum / n;
    }

    /** Cohen kappa: agreement corrected for chance. */
    public static float cohensKappa(int[][] confusion) {
        int n = confusion.length;
        int total = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) total += confusion[i][j];
        if (total == 0) return 0.0f;
        int agree = 0;
        for (int i = 0; i < n; i++) agree += confusion[i][i];
        float po = (float) agree / total;
        float pe = 0.0f;
        for (int i = 0; i < n; i++) {
            int rowSum = 0, colSum = 0;
            for (int j = 0; j < n; j++) {
                rowSum += confusion[i][j];
                colSum += confusion[j][i];
            }
            pe += (float) rowSum * colSum / ((float) total * total);
        }
        if (pe == 1.0f) return 1.0f;
        return (po - pe) / (1.0f - pe);
    }

    /** Matthews correlation coefficient (binary case). */
    public static float matthewsCorrelation(int[][] confusion) {
        if (confusion.length != 2) return 0.0f;
        int tp = confusion[0][0], tn = confusion[1][1];
        int fp = confusion[1][0], fn = confusion[0][1];
        double denom = Math.sqrt((double)(tp + fp) * (tp + fn)
            * (tn + fp) * (tn + fn));
        if (denom == 0.0) return 0.0f;
        return (float)((double)(tp * tn - fp * fn) / denom);
    }

    /** Top-k accuracy. */
    public static float topKAccuracy(float[][] scores, int[] labels, int k) {
        if (scores == null || labels == null
            || scores.length != labels.length || k < 1) {
            throw new IllegalArgumentException("Invalid scores or labels");
        }
        int correct = 0;
        for (int s = 0; s < scores.length; s++) {
            int[] order = argsort(scores[s]);
            for (int i = 0; i < Math.min(k, order.length); i++) {
                if (order[i] == labels[s]) { correct++; break; }
            }
        }
        return (float) correct / scores.length;
    }

    private static int[] argsort(float[] values) {
        Integer[] idx = new Integer[values.length];
        for (int i = 0; i < values.length; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) ->
            Float.compare(values[b], values[a]));
        int[] result = new int[values.length];
        for (int i = 0; i < values.length; i++) result[i] = idx[i];
        return result;
    }
}
