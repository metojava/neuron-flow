package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Position-wise feed-forward network for transformers.
 *
 * <p>Computes FNN(x) = Linear2(GELU(Linear1(x))). The hidden
 * dimension is typically 4 * dModel.</p>
 */
public final class FeedForward {

    private final Linear linear1;
    private final Linear linear2;
    private final int dModel;
    private final int hiddenDim;

    public FeedForward(int dModel, int hiddenDim, long seed) {
        if (dModel <= 0 || hiddenDim <= 0) {
            throw new IllegalArgumentException(
                "dimensions must be positive, got dModel="
                + dModel + ", hiddenDim=" + hiddenDim);
        }
        this.dModel = dModel;
        this.hiddenDim = hiddenDim;
        this.linear1 = new Linear(dModel, hiddenDim, seed);
        this.linear2 = new Linear(hiddenDim, dModel, seed + 1);
    }

    /**
     * Applies the feed-forward transformation to the last
     * dimension of the input.
     *
     * @param x input of shape {..., dModel}
     * @return output of shape {..., dModel}
     */
    public Tensor forward(Tensor x) {
        if (x.size(x.ndim() - 1) != dModel) {
            throw new IllegalArgumentException(
                "expected last dim " + dModel + ", got "
                + x.size(x.ndim() - 1));
        }
        Tensor h = linear1.forward(x);
        h = Ops.gelu(h);
        return linear2.forward(h);
    }

    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(linear1.parameters());
        p.addAll(linear2.parameters());
        return p;
    }
}
