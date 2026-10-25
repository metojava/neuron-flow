package transformers;

import autograd.Ops;
import autograd.Tensor;

/**
 * BERT pooling layer: takes the [CLS] token representation and applies
 * a dense layer with tanh activation.
 */
public final class PoolingLayer {

    private final Linear dense;

    public PoolingLayer(int hiddenSize) {
        this.dense = new Linear(hiddenSize, hiddenSize, 0);
    }

    /**
     * Takes the first token ([CLS]) and applies dense + tanh.
     *
     * @param hiddenStates [B, T, D]
     * @return pooled output [B, D]
     */
    public Tensor forward(Tensor hiddenStates) {
        // Extract first token: [:, 0, :]
        Tensor clsToken = Ops.gatherRows(hiddenStates, new int[]{0});
        Tensor out = dense.forward(clsToken);
        return Ops.tanh(out);
    }

    public java.util.List<autograd.Tensor> parameters() {
        return dense.parameters();
    }
}