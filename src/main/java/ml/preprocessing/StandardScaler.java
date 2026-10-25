package ml.preprocessing;

import ml.ColumnType;
import ml.DataFrame;
import ml.Transformer;

/**
 * Standardises numeric features to zero mean and unit variance:
 * {@code z = (x - mean) / scale}, where {@code scale} is the population
 * standard deviation. Constant features use a scale of {@code 1} so they are
 * merely centred instead of producing a division by zero.
 *
 * <p>Categorical columns pass through untouched, which makes the transformer
 * usable on mixed frames. Missing values must be imputed beforehand.
 */
public final class StandardScaler implements Transformer {

    private double[] means;
    private double[] scales;
    private String[] names;
    private ColumnType[] types;
    private boolean fitted;

    @Override
    public StandardScaler fit(DataFrame X) {
        PreprocessingUtils.requireTrainingFrame(X);
        int columns = X.columns();
        means = new double[columns];
        scales = new double[columns];
        for (int c = 0; c < columns; c++) {
            if (X.type(c) == ColumnType.CATEGORICAL) {
                continue;
            }
            double[] column = X.numericColumn(c);
            double sum = 0.0;
            for (double value : column) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException("Column '" + X.name(c)
                            + "' contains a non-finite value; impute missing values first");
                }
                sum += value;
            }
            double mean = sum / column.length;
            double squared = 0.0;
            for (double value : column) {
                double diff = value - mean;
                squared += diff * diff;
            }
            double variance = squared / column.length;
            means[c] = mean;
            scales[c] = variance > 0.0 ? Math.sqrt(variance) : 1.0;
        }
        names = X.names();
        types = X.types();
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) {
        PreprocessingUtils.requireFitted(fitted, "StandardScaler");
        PreprocessingUtils.requireSameSchema(X, names, types);
        DataFrame.Builder builder = DataFrame.builder(X.rows());
        for (int c = 0; c < names.length; c++) {
            if (types[c] == ColumnType.CATEGORICAL) {
                builder.addCategorical(names[c], X.categoricalColumn(c));
            } else {
                double[] column = X.numericColumn(c);
                for (int r = 0; r < column.length; r++) {
                    if (!Double.isFinite(column[r])) {
                        throw new IllegalArgumentException("Column '" + names[c]
                                + "' contains a non-finite value at row " + r);
                    }
                    column[r] = (column[r] - means[c]) / scales[c];
                }
                builder.addNumeric(names[c], column);
            }
        }
        return builder.build();
    }

    @Override
    public boolean isFitted() {
        return fitted;
    }

    /** Per-column means learned during fitting (zeros for categorical columns). */
    public double[] means() {
        PreprocessingUtils.requireFitted(fitted, "StandardScaler");
        return means.clone();
    }

    /** Per-column standard deviations learned during fitting (ones for categorical columns). */
    public double[] scales() {
        PreprocessingUtils.requireFitted(fitted, "StandardScaler");
        return scales.clone();
    }
}
