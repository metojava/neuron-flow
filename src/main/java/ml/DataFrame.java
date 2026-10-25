package ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Immutable, column-oriented table used by the classical machine-learning API.
 *
 * <p>Columns are either {@link ColumnType#NUMERIC} ({@code double}, with
 * {@link Double#NaN} as the missing marker) or {@link ColumnType#CATEGORICAL}
 * ({@code String}, with {@code null} as the missing marker). Internally each
 * column is stored as a primitive or string array, which keeps feature
 * transformations cache-friendly and avoids boxing during model training.
 *
 * <p>All accessors return defensive copies, so a {@code DataFrame} can be
 * shared safely between preprocessing stages and worker threads.
 */
public final class DataFrame {

    private final String[] names;
    private final ColumnType[] types;
    private final double[][] numericColumns;
    private final String[][] categoricalColumns;
    private final int rows;

    private DataFrame(String[] names, ColumnType[] types,
                      double[][] numericColumns, String[][] categoricalColumns, int rows) {
        this.names = names;
        this.types = types;
        this.numericColumns = numericColumns;
        this.categoricalColumns = categoricalColumns;
        this.rows = rows;
    }

    /**
     * Creates a numeric-only frame.
     *
     * @param names column headers; must be unique
     * @param rows row-major {@code [samples][features]} data
     */
    public static DataFrame ofNumeric(String[] names, double[][] rows) {
        Objects.requireNonNull(names, "names");
        Objects.requireNonNull(rows, "rows");
        if (rows.length == 0) {
            throw new IllegalArgumentException("rows must not be empty");
        }
        int width = rows[0] == null ? -1 : rows[0].length;
        if (width <= 0 || width != names.length) {
            throw new IllegalArgumentException("Row width must match the number of column names");
        }
        for (int r = 1; r < rows.length; r++) {
            if (rows[r] == null || rows[r].length != width) {
                throw new IllegalArgumentException("All rows must have the same width");
            }
        }
        Builder builder = builder(rows.length);
        for (int c = 0; c < width; c++) {
            double[] column = new double[rows.length];
            for (int r = 0; r < rows.length; r++) {
                column[r] = rows[r][c];
            }
            builder.addNumeric(names[c], column);
        }
        return builder.build();
    }

    /**
     * Creates a categorical-only frame.
     *
     * @param names column headers; must be unique
     * @param rows row-major {@code [samples][features]} categorical data
     */
    public static DataFrame ofCategorical(String[] names, String[][] rows) {
        Objects.requireNonNull(names, "names");
        Objects.requireNonNull(rows, "rows");
        if (rows.length == 0) {
            throw new IllegalArgumentException("rows must not be empty");
        }
        int width = rows[0] == null ? -1 : rows[0].length;
        if (width <= 0 || width != names.length) {
            throw new IllegalArgumentException("Row width must match the number of column names");
        }
        for (int r = 1; r < rows.length; r++) {
            if (rows[r] == null || rows[r].length != width) {
                throw new IllegalArgumentException("All rows must have the same width");
            }
        }
        Builder builder = builder(rows.length);
        for (int c = 0; c < width; c++) {
            String[] column = new String[rows.length];
            for (int r = 0; r < rows.length; r++) {
                column[r] = rows[r][c];
            }
            builder.addCategorical(names[c], column);
        }
        return builder.build();
    }

    /** Creates a builder for a mixed numeric/categorical frame with {@code rows} rows. */
    public static Builder builder(int rows) {
        return new Builder(rows);
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return names.length;
    }

    public String[] names() {
        return names.clone();
    }

    public ColumnType[] types() {
        return types.clone();
    }

    public String name(int column) {
        checkColumn(column);
        return names[column];
    }

    public ColumnType type(int column) {
        checkColumn(column);
        return types[column];
    }

    public boolean isNumeric(int column) {
        return type(column) == ColumnType.NUMERIC;
    }

    public boolean isCategorical(int column) {
        return type(column) == ColumnType.CATEGORICAL;
    }

    /** True when every column is numeric, i.e. the frame can feed a model directly. */
    public boolean allNumeric() {
        for (ColumnType type : types) {
            if (type != ColumnType.NUMERIC) {
                return false;
            }
        }
        return true;
    }

    /** Resolves a column name to its index. */
    public int columnIndex(String name) {
        Objects.requireNonNull(name, "name");
        for (int i = 0; i < names.length; i++) {
            if (names[i].equals(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Unknown column: " + name);
    }

    /** Returns a numeric cell; throws if the column is categorical. */
    public double getDouble(int row, int column) {
        checkRow(row);
        checkColumn(column);
        if (types[column] != ColumnType.NUMERIC) {
            throw new IllegalStateException("Column '" + names[column] + "' is categorical");
        }
        return numericColumns[column][row];
    }

    /** Returns a categorical cell; throws if the column is numeric. */
    public String getString(int row, int column) {
        checkRow(row);
        checkColumn(column);
        if (types[column] != ColumnType.CATEGORICAL) {
            throw new IllegalStateException("Column '" + names[column] + "' is numeric");
        }
        return categoricalColumns[column][row];
    }

    /** Returns a defensive copy of a numeric column. */
    public double[] numericColumn(int column) {
        checkColumn(column);
        if (types[column] != ColumnType.NUMERIC) {
            throw new IllegalStateException("Column '" + names[column] + "' is categorical");
        }
        return numericColumns[column].clone();
    }

    /** Returns a defensive copy of a categorical column. */
    public String[] categoricalColumn(int column) {
        checkColumn(column);
        if (types[column] != ColumnType.CATEGORICAL) {
            throw new IllegalStateException("Column '" + names[column] + "' is numeric");
        }
        return categoricalColumns[column].clone();
    }

    /**
     * Converts to a row-major numeric matrix. Categorical columns must be
     * encoded first; otherwise this method fails with a descriptive error.
     */
    public double[][] toDoubleMatrix() {
        for (int c = 0; c < types.length; c++) {
            if (types[c] != ColumnType.NUMERIC) {
                throw new IllegalStateException("Column '" + names[c]
                        + "' is categorical; encode it before model fitting");
            }
        }
        double[][] matrix = new double[rows][names.length];
        for (int c = 0; c < names.length; c++) {
            double[] column = numericColumns[c];
            for (int r = 0; r < rows; r++) {
                matrix[r][c] = column[r];
            }
        }
        return matrix;
    }

    /** Selects columns, preserving their original order and types. */
    public DataFrame select(int... selectedColumns) {
        Objects.requireNonNull(selectedColumns, "selectedColumns");
        if (selectedColumns.length == 0) {
            throw new IllegalArgumentException("At least one column must be selected");
        }
        Builder builder = builder(rows);
        for (int column : selectedColumns) {
            appendTo(builder, column);
        }
        return builder.build();
    }

    /** Selects rows in the supplied order. */
    public DataFrame takeRows(int[] rowIndices) {
        Objects.requireNonNull(rowIndices, "rowIndices");
        if (rowIndices.length == 0) {
            throw new IllegalArgumentException("rowIndices must not be empty");
        }
        Builder builder = builder(rowIndices.length);
        for (int c = 0; c < names.length; c++) {
            if (types[c] == ColumnType.NUMERIC) {
                double[] values = new double[rowIndices.length];
                for (int i = 0; i < rowIndices.length; i++) {
                    values[i] = getDouble(rowIndices[i], c);
                }
                builder.addNumeric(names[c], values);
            } else {
                String[] values = new String[rowIndices.length];
                for (int i = 0; i < rowIndices.length; i++) {
                    values[i] = getString(rowIndices[i], c);
                }
                builder.addCategorical(names[c], values);
            }
        }
        return builder.build();
    }

    /** Appends a numeric column and returns a new frame. */
    public DataFrame appendNumericColumn(String name, double[] values) {
        Builder builder = builder(rows);
        for (int c = 0; c < names.length; c++) {
            appendTo(builder, c);
        }
        builder.addNumeric(name, values);
        return builder.build();
    }

    private void appendTo(Builder builder, int column) {
        checkColumn(column);
        if (types[column] == ColumnType.NUMERIC) {
            builder.addNumeric(names[column], numericColumns[column]);
        } else {
            builder.addCategorical(names[column], categoricalColumns[column]);
        }
    }

    private void checkRow(int row) {
        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException("row " + row + " outside [0, " + rows + ")");
        }
    }

    private void checkColumn(int column) {
        if (column < 0 || column >= names.length) {
            throw new IndexOutOfBoundsException("column " + column + " outside [0, " + names.length + ")");
        }
    }

    @Override
    public String toString() {
        return "DataFrame{rows=" + rows + ", columns=" + Arrays.toString(names)
                + ", types=" + Arrays.toString(types) + '}';
    }

    /** Validating builder for mixed-type frames. */
    public static final class Builder {
        private final int rows;
        private final List<String> names = new ArrayList<>();
        private final List<ColumnType> types = new ArrayList<>();
        private final List<double[]> numeric = new ArrayList<>();
        private final List<String[]> categorical = new ArrayList<>();
        private final Set<String> seenNames = new HashSet<>();

        private Builder(int rows) {
            if (rows < 0) {
                throw new IllegalArgumentException("rows must be non-negative");
            }
            this.rows = rows;
        }

        public Builder addNumeric(String name, double[] values) {
            Objects.requireNonNull(values, "values");
            if (values.length != rows) {
                throw new IllegalArgumentException("Column '" + name + "' has " + values.length
                        + " values but the frame has " + rows + " rows");
            }
            register(name, ColumnType.NUMERIC);
            numeric.add(values.clone());
            categorical.add(null);
            return this;
        }

        public Builder addCategorical(String name, String[] values) {
            Objects.requireNonNull(values, "values");
            if (values.length != rows) {
                throw new IllegalArgumentException("Column '" + name + "' has " + values.length
                        + " values but the frame has " + rows + " rows");
            }
            register(name, ColumnType.CATEGORICAL);
            numeric.add(null);
            categorical.add(values.clone());
            return this;
        }

        public DataFrame build() {
            if (names.isEmpty()) {
                throw new IllegalStateException("A DataFrame must contain at least one column");
            }
            return new DataFrame(names.toArray(new String[0]),
                    types.toArray(new ColumnType[0]),
                    numeric.toArray(new double[0][]),
                    categorical.toArray(new String[0][]),
                    rows);
        }

        private void register(String name, ColumnType type) {
            Objects.requireNonNull(name, "name");
            if (name.isBlank()) {
                throw new IllegalArgumentException("Column name must not be blank");
            }
            if (!seenNames.add(name)) {
                throw new IllegalArgumentException("Duplicate column name: " + name);
            }
            names.add(name);
            types.add(type);
        }
    }
}
