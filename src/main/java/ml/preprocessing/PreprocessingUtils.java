package ml.preprocessing;

import ml.ColumnType;
import ml.DataFrame;

/** Internal helpers shared by preprocessing transformers. */
final class PreprocessingUtils {

    private PreprocessingUtils() {
    }

    static void requireTrainingFrame(DataFrame X) {
        if (X.rows() == 0 || X.columns() == 0) {
            throw new IllegalArgumentException("Transformer requires a non-empty frame");
        }
    }

    /** Ensures a transform input matches the schema seen during fitting. */
    static void requireSameSchema(DataFrame X, String[] names, ColumnType[] types) {
        if (X.columns() != names.length) {
            throw new IllegalArgumentException("Expected " + names.length
                    + " columns but received " + X.columns());
        }
        for (int c = 0; c < names.length; c++) {
            if (!names[c].equals(X.name(c))) {
                throw new IllegalArgumentException("Column " + c + " is '" + X.name(c)
                        + "' but the transformer was fitted on '" + names[c] + "'");
            }
            if (types[c] != X.type(c)) {
                throw new IllegalArgumentException("Column '" + names[c] + "' changed type from "
                        + types[c] + " to " + X.type(c));
            }
        }
    }

    static void requireFitted(boolean fitted, String transformer) {
        if (!fitted) {
            throw new IllegalStateException(transformer + " has not been fitted");
        }
    }
}
