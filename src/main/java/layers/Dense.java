package layers;

import java.util.Arrays;
import java.util.Random;

import vector.VectorMath;

/**
 * Fully connected layer: y = W^T x + b. Weights are stored row-major by
 * input in one flat array (element (i, j) at index i*out + j), so every
 * row is contiguous for SIMD and parameter copies are bulk array copies.
 * Weights use He-normal initialization.
 *
 * <p>Training: {@link #forward} caches its input, {@link #backward}
 * accumulates weight and bias gradients for that input, and
 * {@link #update} applies the summed gradients over a mini-batch.
 */
public class Dense {

    final int in;
    final int out;
    final float[] w;
    final float[] dw;
    final float[] b;
    final float[] db;
    private float[] inputCache;

    /**
     * @param in   number of inputs
     * @param out  number of outputs
     * @param seed RNG seed for weight init
     * @throws IllegalArgumentException if in or out is not positive
     */
    public Dense(int in, int out, long seed) {
        this(in, out, new float[in * out], new float[out]);
        Random rand = new Random(seed);
        float std = (float) Math.sqrt(2.0 / in);
        for (int k = 0; k < w.length; k++) {
            w[k] = (float) (rand.nextGaussian() * std);
        }
    }

    /**
     * Shared-weights constructor: the new layer uses the supplied weight
     * and bias arrays (typically the master's) and owns only its gradient
     * buffers. Used by data-parallel replicas to avoid copying parameters.
     *
     * @param in       number of inputs
     * @param out      number of outputs
     * @param sharedW  shared weight array, length in*out
     * @param sharedB  shared bias array, length out
     */
    public Dense(int in, int out, float[] sharedW, float[] sharedB) {
        if (in <= 0 || out <= 0) {
            throw new IllegalArgumentException("Invalid Dense sizes");
        }
        if (sharedW == null || sharedW.length != in * out) {
            throw new IllegalArgumentException("sharedW must have length " + (in * out));
        }
        if (sharedB == null || sharedB.length != out) {
            throw new IllegalArgumentException("sharedB must have length " + out);
        }
        this.in = in;
        this.out = out;
        this.w = sharedW;
        this.b = sharedB;
        this.dw = new float[in * out];
        this.db = new float[out];
    }

    /**
     * Returns a replica that shares this layer's weight and bias arrays
     * but owns its own gradient buffers.
     *
     * @return a new layer sharing the weights of this one
     */
    public Dense createSharedReplica() {
        return new Dense(in, out, w, b);
    }

    /**
     * @param x input vector of length {@code in}
     * @return output vector of length {@code out}
     */
    public float[] forward(float[] x) {
        if (x.length != in) {
            throw new IllegalArgumentException(
                "Expected " + in + " inputs, got " + x.length);
        }
        inputCache = x;
        float[] result = new float[out];
        System.arraycopy(b, 0, result, 0, out);
        for (int i = 0; i < in; i++) {
            VectorMath.axpyRange(result, 0, x[i], w, i * out, out);
        }
        return result;
    }

    /**
     * Backpropagates through the input cached by the last
     * {@link #forward}, accumulating weight and bias gradients.
     *
     * @param dOut gradient w.r.t. the output, length {@code out}
     * @return gradient w.r.t. the input, length {@code in}
     * @throws IllegalStateException if forward has not been called
     */
    public float[] backward(float[] dOut) {
        if (inputCache == null) {
            throw new IllegalStateException("Call forward before backward");
        }
        float[] dIn = new float[in];
        for (int i = 0; i < in; i++) {
            VectorMath.axpyRange(dw, i * out, inputCache[i], dOut, 0, out);
            dIn[i] = VectorMath.dotRange(w, i * out, dOut, 0, out);
        }
        VectorMath.axpyRange(db, 0, 1.0f, dOut, 0, out);
        return dIn;
    }

    /**
     * Applies the accumulated gradients with plain SGD, averaged over
     * the batch.
     *
     * @param learningRate step size
     * @param batchSize    number of examples summed into the gradients
     */
    public void update(float learningRate, int batchSize) {
        float scale = -learningRate / batchSize;
        VectorMath.axpyRange(w, 0, scale, dw, 0, w.length);
        VectorMath.axpyRange(b, 0, scale, db, 0, out);
    }

    /** Clears the accumulated gradients. */
    public void zeroGrad() {
        Arrays.fill(dw, 0.0f);
        Arrays.fill(db, 0.0f);
    }

    /** @return number of trainable parameters (weights and biases) */
    public int parameterCount() {
        return in * out + out;
    }

    /**
     * Copies weights (row-major by input) then biases into buf.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersTo(float[] buf, int off) {
        System.arraycopy(w, 0, buf, off, w.length);
        System.arraycopy(b, 0, buf, off + w.length, out);
    }

    /**
     * Inverse of {@link #copyParametersTo}.
     *
     * @param buf source, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyParametersFrom(float[] buf, int off) {
        System.arraycopy(buf, off, w, 0, w.length);
        System.arraycopy(buf, off + w.length, b, 0, out);
    }

    /**
     * Copies accumulated weight then bias gradients into buf.
     *
     * @param buf destination, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyGradientsTo(float[] buf, int off) {
        System.arraycopy(dw, 0, buf, off, dw.length);
        System.arraycopy(db, 0, buf, off + dw.length, out);
    }

    /**
     * Inverse of {@link #copyGradientsTo}; overwrites the accumulators.
     *
     * @param buf source, length at least off + parameterCount()
     * @param off starting index
     */
    public void copyGradientsFrom(float[] buf, int off) {
        System.arraycopy(buf, off, dw, 0, dw.length);
        System.arraycopy(buf, off + dw.length, db, 0, out);
    }

    /**
     * Weight at input {@code i} and output {@code j}. Exposed for tests and
     * inspection; do not mutate during training.
     *
     * @param i input index
     * @param j output index
     * @return the weight W[i][j]
     */
    public float weight(int i, int j) {
        return w[i * out + j];
    }
}
