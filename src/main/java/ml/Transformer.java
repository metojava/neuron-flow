package ml;

/**
 * A stateful feature transformation, mirroring scikit-learn's transformer
 * contract: {@link #fit(DataFrame)} learns parameters (means, category
 * dictionaries, ...) from training data and {@link #transform(DataFrame)}
 * applies them, including to unseen data.
 */
public interface Transformer {

    /**
     * Learns the transformation parameters.
     *
     * @return {@code this}, enabling fluent chaining
     */
    Transformer fit(DataFrame X);

    /**
     * Applies the fitted transformation. Implementations must fail with
     * {@link IllegalStateException} when called before {@link #fit(DataFrame)}.
     */
    DataFrame transform(DataFrame X);

    /** Fits and immediately transforms the training data. */
    default DataFrame fitTransform(DataFrame X) {
        return fit(X).transform(X);
    }

    /** Whether this transformer has learned its parameters. */
    boolean isFitted();
}
