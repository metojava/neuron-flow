package ml;

/**
 * A supervised learning algorithm that produces a fitted model from a feature
 * frame and a target vector.
 *
 * @param <M> type of the fitted model
 */
@FunctionalInterface
public interface Estimator<M> {

    /**
     * Fits the model.
     *
     * @param X feature frame; all columns must ultimately be numeric
     * @param y target values, one per row of {@code X}
     * @return fitted model
     */
    M fit(DataFrame X, Vector y);
}
