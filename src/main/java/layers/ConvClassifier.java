package layers;

import activations.Activations;
import activations.Softmax;
import training.TrainableClassifier;

import static activations.Activations.reluBackwardInPlace;
import static activations.Activations.reluInPlace;

/**
 * Convolutional image classifier built from the library's
 * {@link Conv2D}, {@link MaxPool2D}, {@link Dense} layers and the
 * ReLU and softmax helpers in {@link Activations} and {@link Softmax}.
 *
 * <pre>
 *   input 1 x S x S
 *   Conv 8 x 3x3 (pad 1) -&gt; ReLU -&gt; MaxPool 2x2
 *   Conv 16 x 3x3 (pad 1) -&gt; ReLU -&gt; MaxPool 2x2
 *   Flatten -&gt; Dense 128 -&gt; ReLU -&gt; Dense numClasses  (logits)
 * </pre>
 *
 * Images are flat row-major vectors of length S*S. The input side
 * must be at least 4 so that both pooling stages leave a nonempty map.
 *
 * <p>Gradients accumulate over a mini-batch: call {@link #zeroGrad},
 * then {@link #forward} and {@link #backward} per example, then
 * {@link #applyUpdate}.
 */
public final class ConvClassifier implements TrainableClassifier {

    private final int inputSide;
    private final int numClasses;
    private final int secondChannels;
    private final int firstChannels;
    private final int hiddenUnits;
    private final int pooledSide;

    private Conv2D conv1;
    private Conv2D conv2;
    private Dense dense1;
    private Dense dense2;
    private final MaxPool2D pool1 = MaxPool2D.standard();
    private final MaxPool2D pool2 = MaxPool2D.standard();

    private float[][][] c1;
    private float[][][] c2;
    private float[] h1;
    private float[] logits;

    /**
     * Full model: 8 and 16 convolution filters, 128 hidden units.
     *
     * @param seed       RNG seed for reproducible initialization
     * @param inputSide  image height and width S
     * @param numClasses number of output classes, at least 2
     */
    public ConvClassifier(long seed, int inputSide, int numClasses) {
        this(seed, inputSide, numClasses, 8, 16, 128);
    }

    /**
     * Smaller model with 4 and 8 convolution filters and 64 hidden
     * units; the same architecture, trains faster.
     *
     * @param seed       RNG seed for reproducible initialization
     * @param inputSide  image height and width S
     * @param numClasses number of output classes, at least 2
     * @return the compact classifier
     */
    public static ConvClassifier compact(long seed, int inputSide,
                                         int numClasses) {
        return new ConvClassifier(seed, inputSide, numClasses, 4, 8, 64);
    }

    private ConvClassifier(long seed, int inputSide, int numClasses,
                           int firstChannels, int secondChannels,
                           int hiddenUnits) {
        if (inputSide < 4 || numClasses < 2) {
            throw new IllegalArgumentException(
                "Input side must be >= 4 and classes >= 2");
        }
        this.inputSide = inputSide;
        this.numClasses = numClasses;
        this.secondChannels = secondChannels;
        this.firstChannels = firstChannels;
        this.hiddenUnits = hiddenUnits;
        this.pooledSide = pool2.outputSize(pool1.outputSize(inputSide));

        conv1 = new Conv2D(1, firstChannels, 3, 3, 1, 1, seed);
        conv2 = new Conv2D(firstChannels, secondChannels, 3, 3, 1, 1, seed + 1);
        dense1 = new Dense(pooledSide * pooledSide * secondChannels,
                           hiddenUnits, seed + 2);
        dense2 = new Dense(hiddenUnits, numClasses, seed + 3);
    }

    // Package access for ModelLoader.
    Conv2D conv1() {
        return conv1;
    }

    Conv2D conv2() {
        return conv2;
    }

    Dense dense1() {
        return dense1;
    }

    Dense dense2() {
        return dense2;
    }

    /**
     * Forward pass for one flattened image; caches intermediate values
     * for {@link #backward}.
     *
     * @param image flattened row-major pixels, length S*S
     * @return the logits, length numClasses
     * @throws IllegalArgumentException on wrong length or non-finite pixels
     */
    public float[] forward(float[] image) {
        if (image == null || image.length != inputSide * inputSide) {
            throw new IllegalArgumentException(
                "Expected a flattened " + inputSide + "x" + inputSide
                + " image");
        }
        for (float pixel : image) {
            if (!Float.isFinite(pixel)) {
                throw new IllegalArgumentException("Image pixels must be finite");
            }
        }

        float[][][] img = Flatten.unflattenChannelsFirst(
            image, 1, inputSide, inputSide);
        c1 = conv1.forward(img);
        reluInPlace(c1);
        float[][][] p1 = pool1.forward(c1);
        c2 = conv2.forward(p1);
        reluInPlace(c2);
        float[][][] p2 = pool2.forward(c2);
        h1 = dense1.forward(Flatten.flattenChannelsFirst(p2));
        reluInPlace(h1);
        logits = dense2.forward(h1);
        return logits;
    }

    /**
     * Backpropagates the softmax cross-entropy gradient for the example
     * cached by the last {@link #forward}, accumulating gradients.
     *
     * @param label true class index in [0, numClasses)
     * @return the cross-entropy loss for this example
     * @throws IllegalStateException    if forward has not been called
     * @throws IllegalArgumentException if the label is out of range
     */
    public float backward(int label) {
        if (logits == null) {
            throw new IllegalStateException("Call forward before backward");
        }
        if (label < 0 || label >= numClasses) {
            throw new IllegalArgumentException(
                "Label must be in [0, " + (numClasses - 1) + "]");
        }

        float[] probs = Softmax.apply(logits);
        float loss = -(float) Math.log(Math.max(probs[label], 1e-12f));

        float[] d = new float[numClasses];
        for (int j = 0; j < numClasses; j++) {
            d[j] = probs[j] - (j == label ? 1.0f : 0.0f);
        }

        d = dense2.backward(d);
        reluBackwardInPlace(d, h1);
        d = dense1.backward(d);

        float[][][] dP2 = Flatten.unflattenChannelsFirst(
            d, secondChannels, pooledSide, pooledSide);
        float[][][] dR2 = pool2.backward(dP2);
        reluBackwardInPlace(dR2, c2);
        float[][][] dP1 = conv2.backward(dR2);
        float[][][] dR1 = pool1.backward(dP1);
        reluBackwardInPlace(dR1, c1);
        conv1.backward(dR1);

        return loss;
    }

    /**
     * Predicts the class with the highest logit.
     *
     * @param image flattened row-major pixels, length S*S
     * @return predicted class index
     */
    public int predict(float[] image) {
        float[] out = forward(image);
        int best = 0;
        for (int i = 1; i < out.length; i++) {
            if (out[i] > out[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Applies one gradient step to every layer.
     *
     * @param learningRate step size
     * @param batchSize    examples the gradients were summed over
     */
    public void applyUpdate(float learningRate, int batchSize) {
        if (!Float.isFinite(learningRate) || learningRate <= 0.0f
            || batchSize <= 0) {
            throw new IllegalArgumentException(
                "Learning rate and batch size must be positive");
        }
        conv1.update(learningRate, batchSize);
        conv2.update(learningRate, batchSize);
        dense1.update(learningRate, batchSize);
        dense2.update(learningRate, batchSize);
    }

    /** Clears accumulated gradients on every layer. */
    public void zeroGrad() {
        conv1.zeroGrad();
        conv2.zeroGrad();
        dense1.zeroGrad();
        dense2.zeroGrad();
    }

    /**
     * Returns all trainable parameters as one flat vector, in the order
     * conv1, conv2, dense1, dense2. Used to share weights with replicas.
     *
     * @return a new array of length {@link #parameterCount()}
     */
    public float[] getParameters() {
        float[] p = new float[parameterCount()];
        int k = 0;
        conv1.copyParametersTo(p, k);
        k += conv1.parameterCount();
        conv2.copyParametersTo(p, k);
        k += conv2.parameterCount();
        dense1.copyParametersTo(p, k);
        k += dense1.parameterCount();
        dense2.copyParametersTo(p, k);
        return p;
    }

    /**
     * Overwrites all trainable parameters from a flat vector produced by
     * {@link #getParameters()} on a model of the same architecture.
     *
     * @param p flat parameters, length {@link #parameterCount()}
     * @throws IllegalArgumentException on length mismatch
     */
    public void setParameters(float[] p) {
        checkLength(p);
        int k = 0;
        conv1.copyParametersFrom(p, k);
        k += conv1.parameterCount();
        conv2.copyParametersFrom(p, k);
        k += conv2.parameterCount();
        dense1.copyParametersFrom(p, k);
        k += dense1.parameterCount();
        dense2.copyParametersFrom(p, k);
    }

    /**
     * Returns the accumulated gradients as one flat vector in the same
     * layout as {@link #getParameters()}.
     *
     * @return a new array of length {@link #parameterCount()}
     */
    public float[] getGradients() {
        float[] g = new float[parameterCount()];
        int k = 0;
        conv1.copyGradientsTo(g, k);
        k += conv1.parameterCount();
        conv2.copyGradientsTo(g, k);
        k += conv2.parameterCount();
        dense1.copyGradientsTo(g, k);
        k += dense1.parameterCount();
        dense2.copyGradientsTo(g, k);
        return g;
    }

    /**
     * Overwrites the accumulated gradients from a flat vector in the
     * layout of {@link #getGradients()}.
     *
     * @param g flat gradients, length {@link #parameterCount()}
     * @throws IllegalArgumentException on length mismatch
     */
    public void setGradients(float[] g) {
        checkLength(g);
        int k = 0;
        conv1.copyGradientsFrom(g, k);
        k += conv1.parameterCount();
        conv2.copyGradientsFrom(g, k);
        k += conv2.parameterCount();
        dense1.copyGradientsFrom(g, k);
        k += dense1.parameterCount();
        dense2.copyGradientsFrom(g, k);
    }

    private void checkLength(float[] p) {
        if (p == null || p.length != parameterCount()) {
            throw new IllegalArgumentException(
                "Expected " + parameterCount() + " values");
        }
    }

    /** Returns a replica sharing this model's weight arrays. */
    @Override
    public TrainableClassifier createSharedReplica() {
        ConvClassifier r = new ConvClassifier(0L, inputSide, numClasses,
            firstChannels, secondChannels, hiddenUnits);
        // replace layers with shared-weight versions
        r.conv1 = conv1.createSharedReplica();
        r.conv2 = conv2.createSharedReplica();
        r.dense1 = dense1.createSharedReplica();
        r.dense2 = dense2.createSharedReplica();
        return r;
    }

    @Override
    public double accumulateBatch(float[][] images, int[] labels, int[] order,
                                 int from, int to) {
        double loss = 0.0;
        for (int i = from; i < to; i++) {
            int id = order[i];
            forward(images[id]);
            loss += backward(labels[id]);
        }
        return loss;
    }

    /** @return total number of trainable parameters */
    public int parameterCount() {
        return conv1.parameterCount() + conv2.parameterCount()
            + dense1.parameterCount() + dense2.parameterCount();
    }
}
