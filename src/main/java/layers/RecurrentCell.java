package layers;

import optim.Optimizer;

/**
 * Common interface for single-layer recurrent cells ({@link RNN}, {@link LSTM},
 * {@link GRU}). Used by {@link RecurrentStack} to stack layers and directions.
 */
public interface RecurrentCell {

    /**
     * @param sequence [seqLen][inputSize]
     * @return [seqLen][hiddenSize] hidden states
     */
    float[][] forward(float[][] sequence);

    /**
     * Backpropagation through time for the last forward call; accumulates gradients.
     *
     * @param dOutput [seqLen][hiddenSize]
     * @return [seqLen][inputSize]
     */
    float[][] backward(float[][] dOutput);

    int inputSize();

    int hiddenSize();

    int parameterCount();

    void copyParametersTo(float[] buf, int off);

    void copyParametersFrom(float[] buf, int off);

    void copyGradientsTo(float[] buf, int off);

    void zeroGrad();

    void update(float learningRate, int batchSize);

    void update(Optimizer optimizer, int batchSize);
}
