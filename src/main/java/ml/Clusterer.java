package ml;

/**
 * An unsupervised clustering algorithm.
 */
public interface Clusterer {

    /** Learns cluster centres from unlabelled data. */
    Clusterer fit(DataFrame X);

    /** Assigns every row of {@code X} to a zero-based cluster id. */
    int[] predict(DataFrame X);

    /** Convenience for fitting and assigning labels in one call. */
    default int[] fitPredict(DataFrame X) {
        fit(X);
        return predict(X);
    }
}
