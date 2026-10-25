package autograd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A multi-dimensional float tensor with reverse-mode automatic
 * differentiation. Each operation on tensors records a backward
 * closure; calling {@link #backward()} walks the computation
 * graph in reverse topological order and accumulates gradients
 * into every tensor that requires them.
 *
 * <p>Tensors store their data in a flat row-major array. Shape
 * metadata is kept alongside so operations can interpret the
 * flat buffer as any number of dimensions.</p>
 *
 * <p>Gradients accumulate (+=) across multiple backward calls
 * so you can accumulate over mini-batches; call
 * {@link #zeroGrad()} on parameters between optimizer steps.</p>
 */
public final class Tensor {

    private final float[] data;
    private final int[] shape;
    private float[] grad;
    private final boolean requiresGrad;

    private Tensor[] parents;
    private Runnable backwardFn;

    // ── construction ────────────────────────────────────────

    Tensor(float[] data, int[] shape,
           boolean requiresGrad) {
        this.data = data;
        this.shape = shape;
        this.requiresGrad = requiresGrad;
    }

    /**
     * Creates a constant tensor (no gradient tracking).
     *
     * @param data  values in row-major order (copied)
     * @param shape dimensions
     * @return a new tensor
     */
    public static Tensor of(float[] data, int... shape) {
        return new Tensor(data.clone(), shape.clone(), false);
    }

    /**
     * Creates a trainable parameter tensor.
     *
     * @param data  initial values in row-major order (copied)
     * @param shape dimensions
     * @return a new parameter tensor
     */
    public static Tensor param(float[] data, int... shape) {
        return new Tensor(data.clone(), shape.clone(), true);
    }

    /**
     * Creates a zero-filled constant tensor.
     *
     * @param shape dimensions
     * @return a new tensor filled with 0
     */
    public static Tensor zeros(int... shape) {
        return new Tensor(new float[numel(shape)],
            shape.clone(), false);
    }

    /**
     * Creates a trainable parameter initialised to zero.
     *
     * @param shape dimensions
     * @return a new parameter tensor filled with 0
     */
    public static Tensor paramZeros(int... shape) {
        return new Tensor(new float[numel(shape)],
            shape.clone(), true);
    }

    /**
     * Creates a constant tensor sampled from N(0, std).
     *
     * @param std  standard deviation
     * @param seed RNG seed
     * @param shape dimensions
     * @return a new tensor
     */
    public static Tensor randn(float std, long seed,
                               int... shape) {
        return new Tensor(sample(std, seed, numel(shape)),
            shape.clone(), false);
    }

    /**
     * Creates a trainable parameter sampled from N(0, std).
     *
     * @param std  standard deviation
     * @param seed RNG seed
     * @param shape dimensions
     * @return a new parameter tensor
     */
    public static Tensor paramRandn(float std, long seed,
                                    int... shape) {
        return new Tensor(sample(std, seed, numel(shape)),
            shape.clone(), true);
    }

    /**
     * Creates a 1-element constant tensor.
     *
     * @param value the scalar value
     * @return a tensor of shape {1}
     */
    public static Tensor scalar(float value) {
        return new Tensor(new float[]{value},
            new int[]{1}, false);
    }

    // ── accessors ───────────────────────────────────────────

    /** Direct read access to the flat backing array. */
    public float[] data() {
        return data;
    }

    /** Returns a copy of the shape array. */
    public int[] shape() {
        return shape.clone();
    }

    /** Whether this tensor participates in gradient tracking. */
    public boolean requiresGrad() {
        return requiresGrad;
    }

    /** Total number of elements. */
    public int numel() {
        return numel(shape);
    }

    /** Number of dimensions. */
    public int ndim() {
        return shape.length;
    }

    /** Size along a dimension (negative indexes from the end). */
    public int size(int dim) {
        if (dim < 0) dim += shape.length;
        return shape[dim];
    }

    /**
     * Reads a single element.
     *
     * @param indices one index per dimension
     * @return the value at that position
     */
    public float get(int... indices) {
        return data[offset(indices)];
    }

    /**
     * Writes a single element in-place.
     *
     * @param value   new value
     * @param indices one index per dimension
     */
    public void set(float value, int... indices) {
        data[offset(indices)] = value;
    }

    /**
     * Returns the gradient buffer, allocating it (zeros) on
     * first access.
     *
     * @return the flat gradient array (same length as data)
     */
    public float[] grad() {
        if (grad == null) {
            grad = new float[numel()];
        }
        return grad;
    }

    /** Whether a gradient buffer has been allocated. */
    public boolean hasGrad() {
        return grad != null;
    }

    /**
     * Sets the gradient to null so the next backward pass
     * starts fresh.
     */
    public void zeroGrad() {
        grad = null;
    }

    // ── graph wiring (package-private) ──────────────────────

    /**
     * Records the backward closure and parent tensors.
     * Called by ops in this package only.
     */
    void setBackward(Tensor[] parents, Runnable fn) {
        this.parents = parents;
        this.backwardFn = fn;
    }

    // ── backward pass ───────────────────────────────────────

    /**
     * Runs reverse-mode automatic differentiation from this
     * tensor (assumed to be a scalar loss) back through the
     * entire computation graph, accumulating gradients into
     * every parameter that requires them.
     *
     * <p>Gradients accumulate, so call {@link #zeroGrad()} on
     * parameters between optimizer steps unless you
     * intentionally want gradient accumulation.</p>
     *
     * @throws IllegalStateException if this tensor is not a
     *         scalar
     */
    public void backward() {
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        buildTopo(this, visited, topo);
        Collections.reverse(topo);

        float[] g = grad();
        if (g.length != 1) {
            throw new IllegalStateException(
                "backward() requires a scalar tensor, got shape "
                + Arrays.toString(shape));
        }
        g[0] = 1.0f;

        for (Tensor t : topo) {
            if (t.backwardFn != null) {
                t.backwardFn.run();
            }
        }
    }

    private static void buildTopo(Tensor t,
                                  Set<Tensor> visited,
                                  List<Tensor> topo) {
        if (!visited.add(t)) return;
        if (t.parents != null) {
            for (Tensor p : t.parents) {
                buildTopo(p, visited, topo);
            }
        }
        topo.add(t);
    }

    // ── utility ─────────────────────────────────────────────

    /**
     * Returns a tensor that shares this tensor's data but does
     * not require gradients and is not part of the graph.
     *
     * @return a detached copy
     */
    public Tensor detach() {
        return new Tensor(data, shape, false);
    }

    /** Returns a copy of the flat data array. */
    public float[] toFlat() {
        return data.clone();
    }

    private int offset(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                "expected " + shape.length + " indices, got "
                + indices.length);
        }
        int off = 0;
        for (int i = 0; i < shape.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IllegalArgumentException(
                    "index " + indices[i] + " out of bounds for dim "
                    + i + " size " + shape[i]);
            }
            off = off * shape[i] + indices[i];
        }
        return off;
    }

    static int numel(int[] shape) {
        int n = 1;
        for (int s : shape) n *= s;
        return n;
    }

    private static float[] sample(float std, long seed, int n) {
        java.util.Random rand = new java.util.Random(seed);
        float[] d = new float[n];
        for (int i = 0; i < n; i++) {
            d[i] = (float) (rand.nextGaussian() * std);
        }
        return d;
    }

    @Override
    public String toString() {
        return "Tensor(shape=" + Arrays.toString(shape)
            + ", requiresGrad=" + requiresGrad + ")";
    }
}
