package layers;

import java.util.function.Function;

/**
 * A residual (skip) connection: y = F(x) + x, where F is a
 * transform applied to x. The shortcut and the main path must
 * produce tensors of the same shape.
 *
 * <p>Usage:
 * <pre>
 *   ResidualBlock block = new ResidualBlock(x -&gt; relu(conv.forward(x)));
 *   float[][][] out = block.forward(input);
 * </pre>
 */
public final class ResidualBlock {

    private final Function<float[][][], float[][][]> transform;
    private float[][][] inputCache;

    /**
     * @param transform the main path F(x); must return the same shape as its input
     */
    public ResidualBlock(Function<float[][][], float[][][]> transform) {
        if (transform == null) {
            throw new IllegalArgumentException("transform must not be null");
        }
        this.transform = transform;
    }

    /**
     * Forward pass: caches the input, applies the transform, and adds
     * the shortcut.
     *
     * @param input [channels][height][width] feature maps
     * @return [channels][height][width] output (transform(input) + input)
     */
    public float[][][] forward(float[][][] input) {
        if (input == null) {
            throw new IllegalArgumentException("input must not be null");
        }
        inputCache = input;
        float[][][] transformed = transform.apply(input);
        float[][][] out = new float[input.length][][];
        for (int c = 0; c < input.length; c++) {
            out[c] = new float[input[c].length][];
            for (int i = 0; i < input[c].length; i++) {
                out[c][i] = new float[input[c][i].length];
                for (int j = 0; j < input[c][i].length; j++) {
                    out[c][i][j] = transformed[c][i][j] + input[c][i][j];
                }
            }
        }
        return out;
    }

    /** Returns the cached input from the last forward pass. */
    public float[][][] cachedInput() {
        return inputCache;
    }
}
