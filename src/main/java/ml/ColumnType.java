package ml;

/** Logical type of a column in a {@link DataFrame}. */
public enum ColumnType {
    /** Numeric values stored as {@code double}; {@link Double#NaN} denotes a missing value. */
    NUMERIC,
    /** Categorical values stored as {@link String}; {@code null} denotes a missing value. */
    CATEGORICAL
}
