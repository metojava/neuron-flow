package ml;

/**
 * A fitted supervised model capable of point predictions and, for
 * classifiers, per-class probability estimates.
 */
public interface Predictor {

    /**
     * Predicts one value per row of {@code X}.
     *
     * @throws IllegalStateException if the model has not been fitted
     */
    double[] predict(DataFrame X);

    /**
     * Returns class probabilities with shape {@code [rows][classes]}.
     * For binary classifiers the second column is {@code P(class = 1)};
     * for regressors the single column is the point prediction.
     */
    double[][] predictProbabilities(DataFrame X);
}
