package optim;

/**
 * A parameter-update rule. Receives the flattened weight vector
 * and its gradient, updates the weights in place.
 */
public interface Optimizer {

    /**
     * Applies one update step in place.
     *
     * @param parameters all trainable weights and biases,
     *                   modified by this call
     * @param gradients  matching gradient for each parameter
     */
    void update(float[] parameters, float[] gradients);
}
