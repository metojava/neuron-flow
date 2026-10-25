package ml.preprocessing;

import ml.ColumnType;
import ml.DataFrame;
import ml.Transformer;

/**
 * Linearly rescales numeric features into a configurable target range,
 * {@code [0, 1]} by default. Constant features are mapped to the lower bound,
 * matching scikit-learn behaviour. Categorical columns pass through.
 */
public final class MinMaxScaler implements Transformer {

    private final double rangeMin;
    private final double rangeMax;

    private double[] dataMin;
    private double[] dataMax;
    private double[] scales;
    private String[] names;
    private ColumnType[] types;
    private boolean fitted;

    public MinMaxScaler() {
        this(0.0, 1.0);
    }

    public MinMaxScaler(double rangeMin, double rangeMax) {
        if (!Double.isFinite(rangeMin) || !Double.isFinite(rangeMax)) {
            throw new IllegalArgumentException("Range bounds must be finite");
        }
        if (rangeMin >= rangeMax) {
            throw new IllegalArgumentException("rangeMin must be smaller than rangeMax");
        }
        this.rangeMin = rangeMin;
        this.rangeMax = rangeMax;
    }

    @Override
    public MinMaxScaler fit(DataFrame X) {
        PreprocessingUtils.requireTrainingFrame(X);
        int columns = X.columns();
        dataMin = new double[columns];
        dataMax = new double[columns];
        scales = new double[columns];
        double targetSpan = rangeMax - rangeMin;
        for (int c = 0; c < columns; c++) {
            if (X.type(c) == ColumnType.CATEGORICAL) {
                continue;
            }
            double[] column = X.numericColumn(c);
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            for (double value : column) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException("Column '" + X.name(c)
                            + "' contains a non-finite value; impute missing values first");
                }
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            dataMin[c] = min;
            dataMax[c] = max;
            scales[c] = max > min ? targetSpan / (max - min) : 0.0;
        }
        names = X.names();
        types = X.types();
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) {
        PreprocessingUtils.requireFitted(fitted, "MinMaxScaler");
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
                    column[r] = scales[c] == 0.0
                            ? rangeMin
                            : rangeMin + (column[r] - dataMin[c]) * scales[c];
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

    public double rangeMin() {
        return rangeMin;
    }

    public double rangeMax() {
        return rangeMax;
    }

    /** Per-column minima learned during fitting. */
    public double[] dataMin() {
        PreprocessingUtils.requireFitted(fitted, "MinMaxScaler");
        return dataMin.clone();
    }

    /** Per-column maxima learned during fitting. */
    public double[] dataMax() {
        PreprocessingUtils.requireFitted(fitted, "MinMaxScaler");
        return dataMax.clone();
    }
}
