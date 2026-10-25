package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.List;

/**
 * Layer normalisation over the last dimension.
 *
 * <p>Computes out = gamma * (x - mean) / sqrt(var + eps) + beta
 * where mean and variance are computed per-sample across the
 * last dimension only. gamma and beta are learnable parameters
 * of shape {C}.</p>
 */
public final class LayerNorm {

    private final Tensor gamma;
    private final Tensor beta;
    private final int features;

    /**
     * Creates a layer norm initialised to the identity
     * transform (gamma = 1, beta = 0).
     *
     * @param features size of the normalised dimension
     */
    public LayerNorm(int features) {
        if (features <= 0) {
            throw new IllegalArgumentException(
                "features must be positive, got " + features);
        }
        this.features = features;
        float[] g = new float[features];
        java.util.Arrays.fill(g, 1.0f);
        this.gamma = Tensor.param(g, features);
        this.beta = Tensor.paramZeros(features);
    }

    /**
     * Normalises the last dimension of the input.
     *
     * @param x input of shape {..., features}
     * @return normalised tensor, same shape as x
     */
    public Tensor forward(Tensor x) {
        if (x.size(x.ndim() - 1) != features) {
            throw new IllegalArgumentException(
                "expected last dim " + features
                + ", got " + x.size(x.ndim() - 1));
        }
        return Ops.layerNorm(x, gamma, beta);
    }

    /** Returns all trainable parameters of this layer. */
    public List<Tensor> parameters() {
        return List.of(gamma, beta);
    }
}
