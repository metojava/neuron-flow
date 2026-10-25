package layers;

import activations.Activations;
import activations.Softmax;
import optim.Optimizer;
import training.TrainableClassifier;
import vector.VectorMath;

import static activations.Activations.reluBackwardInPlace;
import static activations.Activations.reluInPlace;

/**
 * Fully connected classifier built from the library's {@link Dense}
 * layers. Hidden layers use ReLU; the output layer is linear (logits) and
 * is trained with softmax cross-entropy. For example, sizes {784, 128, 10}
 * is the Keras model Flatten -&gt; Dense(128, relu) -&gt; Dense(10).
 *
 * <p>Parameters are updated either by SGD (each layer's own rule) or by an
 * {@link Optimizer} acting on the flat parameter vector, for example
 * {@link optim.Adam}. The optimizer is held by the model instance, so
 * replicas used by {@link training.DataParallelTrainer} should be created
 * without one.
 */
public final class DenseClassifier implements TrainableClassifier {

    private final int[] sizes;
    private Dense[] layers;
    private final Optimizer optimizer;

    /** Activations of each layer from the last forward pass. */
    private float[][] acts;
    private boolean forwarded;

    /**
     * @param seed      RNG seed for weight initialization
     * @param sizes     layer widths, input first, e.g. {784, 128, 10}
     * @param optimizer optimizer for the flat parameter vector, or null for SGD
     * @throws IllegalArgumentException if fewer than two sizes are given or any is not positive
     */
    public DenseClassifier(long seed, int[] sizes, Optimizer optimizer) {
        if (sizes == null || sizes.length < 2) {
            throw new IllegalArgumentException("Need at least input and output sizes");
        }
        for (int s : sizes) {
            if (s <= 0) {
                throw new IllegalArgumentException("Layer sizes must be positive");
            }
        }
        this.sizes = sizes.clone();
        this.optimizer = optimizer;
        this.layers = new Dense[sizes.length - 1];
        for (int l = 0; l < layers.length; l++) {
            layers[l] = new Dense(sizes[l], sizes[l + 1], seed + l);
        }
        this.acts = new float[layers.length][];
    }

    /** Layers in order; package access for {@link ModelLoader}. */
    Dense[] denseLayers() {
        return layers;
    }

    /** SGD-trained model; see {@link #DenseClassifier(long, int[], Optimizer)}. */
    public DenseClassifier(long seed, int[] sizes) {
        this(seed, sizes, null);
    }

    @Override
    public float[] forward(float[] image) {
        if (image.length != sizes[0]) {
            throw new IllegalArgumentException(
                "Expected " + sizes[0] + " inputs, got " + image.length);
        }
        float[] x = image;
        for (int l = 0; l < layers.length; l++) {
            float[] out = layers[l].forward(x);
            if (l < layers.length - 1) {
                reluInPlace(out);
            }
            acts[l] = out;
            x = out;
        }
        forwarded = true;
        return x;
    }

    @Override
    public float backward(int label) {
        if (!forwarded) {
            throw new IllegalStateException("Call forward before backward");
        }
        if (label < 0 || label >= sizes[sizes.length - 1]) {
            throw new IllegalArgumentException("Label out of range");
        }
        float[] probs = Softmax.apply(acts[layers.length - 1]);
        float loss = -(float) Math.log(Math.max(probs[label], 1e-12f));

        float[] d = probs;
        d[label] -= 1.0f;   // softmax cross-entropy gradient: p - one-hot

        for (int l = layers.length - 1; l >= 0; l--) {
            d = layers[l].backward(d);
            if (l > 0) {
                reluBackwardInPlace(d, acts[l - 1]);
            }
        }
        return loss;
    }

    @Override
    public int predict(float[] image) {
        return argmax(forward(image));
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    @Override
    public void zeroGrad() {
        for (Dense layer : layers) {
            layer.zeroGrad();
        }
    }

    @Override
    public void applyUpdate(float learningRate, int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        if (optimizer == null) {
            for (Dense layer : layers) {
                layer.update(learningRate, batchSize);
            }
            return;
        }
        float[] params = getParameters();
        float[] grads = getGradients();
        float scale = 1.0f / batchSize;
        for (int k = 0; k < grads.length; k++) {
            grads[k] *= scale;
        }
        optimizer.update(params, grads);
        setParameters(params);
    }

    @Override
    public int parameterCount() {
        int total = 0;
        for (Dense layer : layers) {
            total += layer.parameterCount();
        }
        return total;
    }

    @Override
    public float[] getParameters() {
        float[] p = new float[parameterCount()];
        int k = 0;
        for (Dense layer : layers) {
            layer.copyParametersTo(p, k);
            k += layer.parameterCount();
        }
        return p;
    }

    @Override
    public void setParameters(float[] p) {
        checkLength(p);
        int k = 0;
        for (Dense layer : layers) {
            layer.copyParametersFrom(p, k);
            k += layer.parameterCount();
        }
    }

    @Override
    public float[] getGradients() {
        float[] g = new float[parameterCount()];
        int k = 0;
        for (Dense layer : layers) {
            layer.copyGradientsTo(g, k);
            k += layer.parameterCount();
        }
        return g;
    }

    @Override
    public void setGradients(float[] g) {
        checkLength(g);
        int k = 0;
        for (Dense layer : layers) {
            layer.copyGradientsFrom(g, k);
            k += layer.parameterCount();
        }
    }

    @Override
    public TrainableClassifier createSharedReplica() {
        DenseClassifier r = new DenseClassifier(0L, sizes.clone(), null);
        for (int l = 0; l < layers.length; l++) {
            r.layers[l] = layers[l].createSharedReplica();
        }
        return r;
    }

    @Override
    public double accumulateBatch(float[][] images, int[] labels, int[] order,
                                 int from, int to) {
        int n = to - from;
        int in = sizes[0];
        int hidden = sizes[1];
        int out = sizes[2];

        // Forward pass: layer 1 (ReLU) then layer 2 (linear)
        float[][] y1 = new float[n][hidden];
        for (int s = 0; s < n; s++) {
            System.arraycopy(layers[0].b, 0, y1[s], 0, hidden);
        }
        for (int i = 0; i < in; i++) {
            for (int s = 0; s < n; s++) {
                VectorMath.axpyRange(y1[s], 0, images[order[from + s]][i],
                    layers[0].w, i * hidden, hidden);
            }
        }
        for (int s = 0; s < n; s++) {
            Activations.reluInPlace(y1[s]);
        }

        float[][] y2 = new float[n][out];
        for (int s = 0; s < n; s++) {
            System.arraycopy(layers[1].b, 0, y2[s], 0, out);
        }
        for (int i = 0; i < hidden; i++) {
            for (int s = 0; s < n; s++) {
                VectorMath.axpyRange(y2[s], 0, y1[s][i],
                    layers[1].w, i * out, out);
            }
        }

        // Loss and output gradient
        double totalLoss = 0.0;
        float[][] dY2 = new float[n][out];
        for (int s = 0; s < n; s++) {
            float[] probs = Softmax.apply(y2[s]);
            int label = labels[order[from + s]];
            totalLoss -= Math.log(Math.max(probs[label], 1e-12f));
            for (int j = 0; j < out; j++) {
                dY2[s][j] = probs[j] - (j == label ? 1.0f : 0.0f);
            }
        }

        // Backward pass: layer 2 gradients
        for (int i = 0; i < hidden; i++) {
            for (int j = 0; j < out; j++) {
                float sum = 0.0f;
                for (int s = 0; s < n; s++) {
                    sum += y1[s][i] * dY2[s][j];
                }
                layers[1].dw[i * out + j] += sum;
            }
        }
        for (int j = 0; j < out; j++) {
            float sum = 0.0f;
            for (int s = 0; s < n; s++) {
                sum += dY2[s][j];
            }
            layers[1].db[j] += sum;
        }

        // Backward pass: layer 1 gradients (dY1 = dY2 @ W2^T, then ReLU)
        float[][] dY1 = new float[n][hidden];
        for (int s = 0; s < n; s++) {
            for (int i = 0; i < hidden; i++) {
                dY1[s][i] = VectorMath.dotRange(
                    layers[1].w, i * out, dY2[s], 0, out);
            }
        }
        for (int s = 0; s < n; s++) {
            for (int i = 0; i < hidden; i++) {
                if (y1[s][i] <= 0.0f) {
                    dY1[s][i] = 0.0f;
                }
            }
        }

        // dW1 = X^T @ dY1, db1 = sum(dY1)
        for (int i = 0; i < in; i++) {
            for (int j = 0; j < hidden; j++) {
                float sum = 0.0f;
                for (int s = 0; s < n; s++) {
                    sum += images[order[from + s]][i] * dY1[s][j];
                }
                layers[0].dw[i * hidden + j] += sum;
            }
        }
        for (int j = 0; j < hidden; j++) {
            float sum = 0.0f;
            for (int s = 0; s < n; s++) {
                sum += dY1[s][j];
            }
            layers[0].db[j] += sum;
        }

        return totalLoss;
    }

    private void checkLength(float[] v) {
        if (v == null || v.length != parameterCount()) {
            throw new IllegalArgumentException(
                "Expected " + parameterCount() + " values");
        }
    }
}
