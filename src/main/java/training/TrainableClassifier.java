package training;

/**
 * A classifier whose parameters and gradients can be read and written as
 * flat vectors. Lets {@link DataParallelTrainer} train any model that
 * implements it.
 */
public interface TrainableClassifier {

    /**
     * Forward pass for one flattened image; caches values for
     * {@link #backward(int)}.
     *
     * @param image flattened input
     * @return output scores (logits)
     */
    float[] forward(float[] image);

    /**
     * Backward pass for the example cached by the last forward call.
     * Adds this example's gradients to the accumulators.
     *
     * @param label true class index
     * @return cross-entropy loss for this example
     */
    float backward(int label);

    /**
     * Predicted class: the index of the largest output score.
     *
     * @param image flattened input
     * @return class index
     */
    int predict(float[] image);

    /** Clears the gradient accumulators. */
    void zeroGrad();

    /**
     * Applies the accumulated gradients, averaged over {@code batchSize}
     * examples.
     *
     * @param learningRate step size (implementations may ignore it when
     *                     they hold their own optimizer)
     * @param batchSize    number of examples summed into the gradients
     */
    void applyUpdate(float learningRate, int batchSize);

    /** @return number of trainable parameters */
    int parameterCount();

    /** @return all parameters as one flat vector */
    float[] getParameters();

    /** @param p flat parameters, length {@link #parameterCount()} */
    void setParameters(float[] p);

    /** @return accumulated gradients as one flat vector */
    float[] getGradients();

    /** @param g flat gradients, length {@link #parameterCount()} */
    void setGradients(float[] g);

    /**
     * Processes a contiguous shard [from, to) of a batch: forward
     * and backward for every sample, accumulating gradients. Does not
     * apply the update.
     *
     * @param images  all images, indexed by sample id
     * @param labels  class label per sample id
     * @param order   sample ids in visiting order
     * @param from    first index into order for this shard
     * @param to      one past the last index into order for this shard
     * @return mean cross-entropy loss over the shard
     */
    double accumulateBatch(float[][] images, int[] labels, int[] order,
                           int from, int to);

    /**
     * Returns a new classifier of the same architecture that shares this
     * instance's weight arrays (read-only during a batch) but owns its own
     * gradient buffers. Used by data-parallel training to avoid copying
     * parameters between the master and its replicas.
     *
     * @return a shared-weight replica
     */
    TrainableClassifier createSharedReplica();
}
