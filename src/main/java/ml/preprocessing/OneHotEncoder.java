package ml.preprocessing;

import ml.ColumnType;
import ml.DataFrame;
import ml.Transformer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * Expands categorical columns into binary indicator columns named
 * {@code column[category]}. Numeric columns pass through unchanged.
 *
 * <p>By default the first category of every feature is dropped to avoid
 * perfect collinearity in linear models, and previously unseen categories are
 * encoded as all zeros at transform time. Both behaviours are configurable.
 * Missing categories ({@code null}) should normally be handled by a
 * {@link SimpleImputer} placed earlier in the pipeline.
 */
public final class OneHotEncoder implements Transformer {

    /** Behaviour when a transform input contains a category not seen during fitting. */
    public enum UnknownStrategy {
        ERROR,
        IGNORE
    }

    private final boolean dropFirst;
    private final UnknownStrategy unknownStrategy;

    private String[][] categories;
    private Map<String, Integer>[] categoryIndex;
    private String[] names;
    private ColumnType[] types;
    private boolean fitted;

    public OneHotEncoder() {
        this(true, UnknownStrategy.IGNORE);
    }

    public OneHotEncoder(boolean dropFirst, UnknownStrategy unknownStrategy) {
        this.dropFirst = dropFirst;
        this.unknownStrategy = unknownStrategy;
    }

    @Override
    @SuppressWarnings("unchecked")
    public OneHotEncoder fit(DataFrame X) {
        PreprocessingUtils.requireTrainingFrame(X);
        int columns = X.columns();
        categories = new String[columns][];
        categoryIndex = new Map[columns];
        for (int c = 0; c < columns; c++) {
            if (X.type(c) != ColumnType.CATEGORICAL) {
                continue;
            }
            TreeSet<String> distinct = new TreeSet<>();
            for (String value : X.categoricalColumn(c)) {
                if (value != null) {
                    distinct.add(value);
                }
            }
            categories[c] = distinct.toArray(new String[0]);
            Map<String, Integer> index = new HashMap<>();
            for (int i = 0; i < categories[c].length; i++) {
                index.put(categories[c][i], i);
            }
            categoryIndex[c] = index;
        }
        names = X.names();
        types = X.types();
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) {
        PreprocessingUtils.requireFitted(fitted, "OneHotEncoder");
        PreprocessingUtils.requireSameSchema(X, names, types);
        DataFrame.Builder builder = DataFrame.builder(X.rows());
        int emittedColumns = 0;
        for (int c = 0; c < names.length; c++) {
            if (types[c] == ColumnType.NUMERIC) {
                builder.addNumeric(names[c], X.numericColumn(c));
                emittedColumns++;
                continue;
            }
            String[] values = X.categoricalColumn(c);
            int first = dropFirst ? 1 : 0;
            for (int category = first; category < categories[c].length; category++) {
                String outputName = names[c] + "[" + categories[c][category] + "]";
                double[] indicator = new double[X.rows()];
                for (int r = 0; r < values.length; r++) {
                    Integer index = values[r] == null ? null : categoryIndex[c].get(values[r]);
                    if (index == null) {
                        if (unknownStrategy == UnknownStrategy.ERROR) {
                            throw new IllegalArgumentException("Unknown category '" + values[r]
                                    + "' in column '" + names[c] + "' at row " + r);
                        }
                    } else if (index == category) {
                        indicator[r] = 1.0;
                    }
                }
                builder.addNumeric(outputName, indicator);
                emittedColumns++;
            }
        }
        if (emittedColumns == 0) {
            throw new IllegalStateException("OneHotEncoder produced no columns; disable dropFirst "
                    + "or fit on data with more than one category per column");
        }
        return builder.build();
    }

    @Override
    public boolean isFitted() {
        return fitted;
    }

    public boolean dropFirst() {
        return dropFirst;
    }

    public UnknownStrategy unknownStrategy() {
        return unknownStrategy;
    }

    /** Learned categories for an input column; empty for numeric columns. */
    public String[] categories(int column) {
        PreprocessingUtils.requireFitted(fitted, "OneHotEncoder");
        if (column < 0 || column >= names.length) {
            throw new IndexOutOfBoundsException("column " + column + " outside [0, " + names.length + ")");
        }
        return categories[column] == null ? new String[0] : categories[column].clone();
    }

    /** Category dictionaries grouped by input column, useful for inspection. */
    public List<String> categoriesOf(String columnName) {
        PreprocessingUtils.requireFitted(fitted, "OneHotEncoder");
        for (int c = 0; c < names.length; c++) {
            if (names[c].equals(columnName)) {
                return categories[c] == null ? List.of() : List.of(categories[c]);
            }
        }
        throw new IllegalArgumentException("Unknown column: " + columnName);
    }

    /** Indicator column names that {@link #transform(DataFrame)} produces for a feature. */
    public List<String> outputNames(String columnName) {
        PreprocessingUtils.requireFitted(fitted, "OneHotEncoder");
        for (int c = 0; c < names.length; c++) {
            if (!names[c].equals(columnName)) {
                continue;
            }
            if (types[c] == ColumnType.NUMERIC) {
                return List.of(columnName);
            }
            List<String> result = new ArrayList<>();
            int first = dropFirst ? 1 : 0;
            for (int i = first; i < categories[c].length; i++) {
                result.add(columnName + "[" + categories[c][i] + "]");
            }
            return List.copyOf(result);
        }
        throw new IllegalArgumentException("Unknown column: " + columnName);
    }
}
