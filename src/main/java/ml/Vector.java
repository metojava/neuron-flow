package ml;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable dense vector of primitive {@code double} values.
 *
 * <p>Targets and predictions use this type. For classification the values are
 * class labels; for regression they are continuous responses. Class labels are
 * doubles so that any numeric encoding (0/1, -1/+1, arbitrary ids) is accepted.
 */
public final class Vector {

    private final double[] values;

    private Vector(double[] values) {
        this.values = values;
    }

    /** Creates a vector from the supplied values; the array is copied. */
    public static Vector of(double... values) {
        Objects.requireNonNull(values, "values");
        return new Vector(values.clone());
    }

    /** Creates a vector from an array already owned by the caller (still copied defensively). */
    public static Vector copyOf(double[] values) {
        return of(values);
    }

    /** Creates a vector of {@code size} zeros. */
    public static Vector zeros(int size) {
        if (size < 0) {
            throw new IllegalArgumentException("size must be non-negative");
        }
        return new Vector(new double[size]);
    }

    /** Creates a vector by selecting {@code indices} from this vector. */
    public Vector take(int[] indices) {
        Objects.requireNonNull(indices, "indices");
        double[] result = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            result[i] = get(indices[i]);
        }
        return new Vector(result);
    }

    public int size() {
        return values.length;
    }

    public double get(int index) {
        if (index < 0 || index >= values.length) {
            throw new IndexOutOfBoundsException("index " + index + " outside [0, " + values.length + ")");
        }
        return values[index];
    }

    /** Returns a defensive copy of the backing values. */
    public double[] toArray() {
        return values.clone();
    }

    /** Number of occurrences of {@code value}. */
    public int count(double value) {
        int count = 0;
        for (double current : values) {
            if (Double.compare(current, value) == 0) {
                count++;
            }
        }
        return count;
    }

    /** Sorted distinct values, using {@link Double#compare(double, double)} ordering. */
    public double[] uniqueSorted() {
        return Arrays.stream(values).distinct().sorted().toArray();
    }

    /** Arithmetic mean, ignoring {@link Double#NaN} values. */
    public double mean() {
        double sum = 0.0;
        int count = 0;
        for (double value : values) {
            if (!Double.isNaN(value)) {
                sum += value;
                count++;
            }
        }
        return count == 0 ? Double.NaN : sum / count;
    }

    /** @throws IllegalArgumentException if any value is not finite */
    public void requireFinite(String context) {
        for (int i = 0; i < values.length; i++) {
            if (!Double.isFinite(values[i])) {
                throw new IllegalArgumentException(context + " contains a non-finite value at index " + i);
            }
        }
    }

    @Override
    public String toString() {
        return Arrays.toString(values);
    }
}
