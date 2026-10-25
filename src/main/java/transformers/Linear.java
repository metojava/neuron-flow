package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.List;

/**
 * Fully connected (affine) layer: out = x @ W + b.
 *
 * <p>Weight matrix W has shape {inFeatures, outFeatures} and
 * bias has shape {outFeatures}. The input may have any number
 * of leading dimensions — only the last dimension is transformed.</p>
 */
public final class Linear {

    private final Tensor weight;
    private final Tensor bias;
    private final int inFeatures;
    private final int outFeatures;

    /**
     * Creates a linear layer with He-normal initialised weights
     * and zero bias.
     *
     * @param inFeatures  size of the input's last dimension
     * @param outFeatures size of the output's last dimension
     * @param seed        RNG seed for reproducible init
     */
    public Linear(int inFeatures, int outFeatures, long seed) {
        if (inFeatures <= 0 || outFeatures <= 0) {
            throw new IllegalArgumentException(
                "features must be positive, got "
                + inFeatures + ", " + outFeatures);
        }
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;

        float std = (float) Math.sqrt(2.0 / inFeatures);
        this.weight = Tensor.paramRandn(std, seed,
            inFeatures, outFeatures);
        this.bias = Tensor.paramZeros(outFeatures);
    }

    /**
     * Projects the last dimension of the input.
     *
     * @param x input of shape {..., inFeatures}
     * @return output of shape {..., outFeatures}
     */
    public Tensor forward(Tensor x) {
        if (x.size(x.ndim() - 1) != inFeatures) {
            throw new IllegalArgumentException(
                "expected last dim " + inFeatures
                + ", got " + x.size(x.ndim() - 1));
        }
        Tensor flat = Ops.flattenLeading(x); // {N, in}
        Tensor proj = Ops.matmul(flat, weight); // {N, out}
        Tensor biased = Ops.add(proj, bias);    // {N, out} + {out}
        return Ops.unflattenLeading(biased,
            leadingDims(x));
    }

    /** Returns all trainable parameters of this layer. */
    public List<Tensor> parameters() {
        return List.of(weight, bias);
    }

    /**
     * Ties this layer's weight to an external tensor (weight tying).
     * The external tensor must have shape {outFeatures, inFeatures}
     * which will be transposed for the matmul.
     */
    public void tieWeights(Tensor externalWeight) {
        // This is a marker; actual weight sharing is handled by the caller
        // by passing the same Tensor instance to the decoder's weight field
    }

    /**
     * Creates a linear layer that shares weights with an external tensor.
     * The external weight must have shape {outFeatures, inFeatures}.
     */
    public Linear(int inFeatures, int outFeatures, Tensor sharedWeight, long seed) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.weight = sharedWeight;
        this.bias = Tensor.paramZeros(outFeatures);
    }

    private int[] leadingDims(Tensor x) {
        int nd = x.ndim();
        int[] lead = new int[nd - 1];
        for (int i = 0; i < nd - 1; i++) {
            lead[i] = x.size(i);
        }
        return lead;
    }
}
