package ml.preprocessing;

import ml.ColumnType;
import ml.DataFrame;
import ml.Transformer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Replaces missing values with a learned statistic.
 *
 * <p>Numeric columns use {@link Double#NaN} as the missing marker and can be
 * imputed with the mean, median or most frequent value. Categorical columns
 * use {@code null} as the missing marker and always use the most frequent
 * value, because mean and median are undefined for them. A column whose
 * values are all missing is filled with {@code 0} (numeric) or
 * {@code "__missing__"} (categorical).
 */
public final class SimpleImputer implements Transformer {

    /** Supported imputation statistics. */
    public enum Strategy {
        MEAN,
        MEDIAN,
        MOST_FREQUENT
    }

    private static final String EMPTY_CATEGORY = "__missing__";

    private final Strategy strategy;
    private double[] numericFill;
    private String[] categoricalFill;
    private String[] names;
    private ColumnType[] types;
    private boolean fitted;

    public SimpleImputer() {
        this(Strategy.MEAN);
    }

    public SimpleImputer(Strategy strategy) {
        this.strategy = strategy;
    }

    @Override
    public SimpleImputer fit(DataFrame X) {
        PreprocessingUtils.requireTrainingFrame(X);
        int columns = X.columns();
        numericFill = new double[columns];
        categoricalFill = new String[columns];
        for (int c = 0; c < columns; c++) {
            if (X.type(c) == ColumnType.NUMERIC) {
                numericFill[c] = numericStatistic(X.numericColumn(c));
            } else {
                categoricalFill[c] = mostFrequentCategory(X.categoricalColumn(c));
            }
        }
        names = X.names();
        types = X.types();
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) {
        PreprocessingUtils.requireFitted(fitted, "SimpleImputer");
        PreprocessingUtils.requireSameSchema(X, names, types);
        DataFrame.Builder builder = DataFrame.builder(X.rows());
        for (int c = 0; c < names.length; c++) {
            if (types[c] == ColumnType.NUMERIC) {
                double[] column = X.numericColumn(c);
                for (int r = 0; r < column.length; r++) {
                    if (Double.isNaN(column[r])) {
                        column[r] = numericFill[c];
                    }
                }
                builder.addNumeric(names[c], column);
            } else {
                String[] column = X.categoricalColumn(c);
                for (int r = 0; r < column.length; r++) {
                    if (column[r] == null) {
                        column[r] = categoricalFill[c];
                    }
                }
                builder.addCategorical(names[c], column);
            }
        }
        return builder.build();
    }

    @Override
    public boolean isFitted() {
        return fitted;
    }

    public Strategy strategy() {
        return strategy;
    }

    /** Learned numeric fill values; zeros for categorical columns. */
    public double[] numericStatistics() {
        PreprocessingUtils.requireFitted(fitted, "SimpleImputer");
        return numericFill.clone();
    }

    private double numericStatistic(double[] column) {
        if (strategy == Strategy.MOST_FREQUENT) {
            return mostFrequentNumber(column);
        }
        double[] present = Arrays.stream(column).filter(Double::isFinite).sorted().toArray();
        if (present.length == 0) {
            return 0.0;
        }
        if (strategy == Strategy.MEAN) {
            double sum = 0.0;
            for (double value : present) {
                sum += value;
            }
            return sum / present.length;
        }
        int middle = present.length / 2;
        return present.length % 2 == 1
                ? present[middle]
                : (present[middle - 1] + present[middle]) / 2.0;
    }

    private static double mostFrequentNumber(double[] column) {
        Map<Double, Integer> counts = new HashMap<>();
        double best = 0.0;
        int bestCount = 0;
        for (double value : column) {
            if (!Double.isFinite(value)) {
                continue;
            }
            int count = counts.merge(value, 1, Integer::sum);
            if (count > bestCount || (count == bestCount && value < best)) {
                best = value;
                bestCount = count;
            }
        }
        return bestCount == 0 ? 0.0 : best;
    }

    private static String mostFrequentCategory(String[] column) {
        Map<String, Integer> counts = new HashMap<>();
        String best = null;
        int bestCount = 0;
        for (String value : column) {
            if (value == null) {
                continue;
            }
            int count = counts.merge(value, 1, Integer::sum);
            if (count > bestCount || (count == bestCount && best != null && value.compareTo(best) < 0)) {
                best = value;
                bestCount = count;
            }
        }
        return bestCount == 0 ? EMPTY_CATEGORY : best;
    }
}
