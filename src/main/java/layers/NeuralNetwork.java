package layers;

import activations.ActivationFunction;
import initialization.WeightInitializer;
import losses.LossFunction;
import optim.Optimizer;
import optim.SGD;
import training.Trainer;
import vector.VectorMath;

import java.util.Random;

/**
 * A fully connected feed-forward neural network built from
 * {@link Layer}s. Implements forward propagation, loss
 * calculation, backpropagation with gradient accumulation, and
 * mini-batch training driven by an {@link Optimizer}.
 * All computation uses float precision. The inner loops of the
 * forward and backward passes are SIMD-accelerated through
 * {@link VectorMath} (JDK incubating Vector API).
 */
public class NeuralNetwork {

    private final Layer[] layers;
    private final ActivationFunction[] activations;
    private final LossFunction lossFunction;
    private final Optimizer optimizer;

    /** Reusable buffer holding the previous layer's values for
     *  SIMD dot products; sized to the widest layer. */
    private final float[] prevBuffer;

    /** Reusable buffer for backpropagated delta accumulation;
     *  sized to the widest hidden layer. */
    private final float[] deltaBuffer;

    /** Dropout rate per layer (0 = no dropout). */
    private final float[] dropoutRates;

    /** Whether the network is in training mode (dropout active). */
    private boolean training = true;

    /** Dropout masks from the last forward pass, one per layer. */
    private boolean[][] dropoutMasks;

    /**
     * Convenience constructor: linear input layer, sigmoid on
     * every other layer, mean squared error and plain SGD.
     *
     * @param topology     layer sizes, e.g. {784, 128, 10}
     * @param learningRate SGD step size
     */
    public NeuralNetwork(int[] topology, float learningRate) {
        this(topology, repeatedSigmoid(topology),
             LossFunction.meanSquaredError(),
             new SGD(learningRate), 42L);
    }

    /**
     * Creates a network with Xavier-uniform weight scaling.
     *
     * @param topology      layer sizes, index 0 is the input layer
     * @param activations   one activation per layer
     * @param lossFunction  error metric used during training
     * @param optimizer     update rule for weights
     * @param seed          RNG seed for reproducibility
     * @throws IllegalArgumentException on invalid configuration
     */
    public NeuralNetwork(
            int[] topology,
            ActivationFunction[] activations,
            LossFunction lossFunction,
            Optimizer optimizer,
            long seed) {
        this(topology, activations, lossFunction, optimizer,
             seed, WeightInitializer.xavierUniform());
    }

    /**
     * Full constructor with an explicit weight initializer.
     *
     * @param topology      layer sizes, index 0 is the input layer
     * @param activations   one activation per layer
     * @param lossFunction  error metric used during training
     * @param optimizer     update rule for weights
     * @param seed          RNG seed for reproducibility
     * @param initializer   weight init strategy (e.g. He normal
     *                      for ReLU networks)
     * @throws IllegalArgumentException on invalid configuration
     */
    public NeuralNetwork(
            int[] topology,
            ActivationFunction[] activations,
            LossFunction lossFunction,
            Optimizer optimizer,
            long seed,
            WeightInitializer initializer) {

        if (topology == null || topology.length < 2) {
            throw new IllegalArgumentException(
                "Topology must contain input and output layers");
        }

        for (int size : topology) {
            if (size <= 0) {
                throw new IllegalArgumentException(
                    "Layer sizes must be positive");
            }
        }

        if (activations == null ||
            activations.length != topology.length ||
            lossFunction == null || optimizer == null ||
            initializer == null) {
            throw new IllegalArgumentException(
                "Invalid network configuration");
        }

        this.activations = activations.clone();
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.layers = new Layer[topology.length];
        this.dropoutRates = new float[topology.length];

        int maxSize = 0;
        for (int size : topology) {
            if (size > maxSize) {
                maxSize = size;
            }
        }
        this.prevBuffer = new float[maxSize];
        this.deltaBuffer = new float[maxSize];
        this.dropoutMasks = new boolean[topology.length][];

        Random rand = new Random(seed);
        layers[0] = new Layer(topology[0], 0, rand);

        for (int l = 1; l < topology.length; l++) {
            int fanIn = topology[l - 1];
            int fanOut = topology[l];
            layers[l] = new Layer(topology[l], fanIn, rand);

            for (Neuron neuron : layers[l].neurons) {
                initializer.initialize(
                    neuron.weights, fanIn, fanOut, rand);
                neuron.bias = 0.0f;
            }
        }
    }

    private static ActivationFunction[] repeatedSigmoid(
            int[] topology) {
        if (topology == null || topology.length < 2) {
            throw new IllegalArgumentException("Invalid topology");
        }

        ActivationFunction[] result =
            new ActivationFunction[topology.length];

        result[0] = ActivationFunction.linear();

        for (int i = 1; i < result.length; i++) {
            result[i] = ActivationFunction.sigmoid();
        }

        return result;
    }

    /**
     * Runs one input vector through the network.
     * Layer l computes z = bias + sum(prev.value * weight) and
     * stores activation[l](z) as the neuron's value; the last
     * layer's values form the returned vector.
     *
     * @param inputs one example, length = input layer size
     * @return one output per output-layer neuron
     * @throws IllegalArgumentException on wrong length or
     *                                  non-finite inputs
     */
    public float[] forwardPropagate(float[] inputs) {
        if (inputs == null ||
            inputs.length != layers[0].neurons.length) {
            throw new IllegalArgumentException(
                "Input dimension does not match network");
        }

        for (int i = 0; i < inputs.length; i++) {
            if (!Float.isFinite(inputs[i])) {
                throw new IllegalArgumentException(
                    "Inputs must be finite");
            }

            layers[0].neurons[i].value = inputs[i];
        }

        for (int l = 1; l < layers.length; l++) {
            Layer previous = layers[l - 1];
            Layer current = layers[l];

            int prevLen = previous.neurons.length;
            for (int j = 0; j < prevLen; j++) {
                prevBuffer[j] = previous.neurons[j].value;
            }

            for (Neuron neuron : current.neurons) {
                float sum = neuron.bias
                    + VectorMath.dot(prevBuffer, neuron.weights,
                                     prevLen);
                neuron.preActivation = sum;
                neuron.value = activations[l].activate(sum);
            }

            // Apply dropout
            if (training && dropoutRates[l] > 0.0f) {
                float rate = dropoutRates[l];
                float scale = 1.0f / (1.0f - rate);
                if (dropoutMasks[l] == null ||
                    dropoutMasks[l].length != current.neurons.length) {
                    dropoutMasks[l] = new boolean[current.neurons.length];
                }
                for (int i = 0; i < current.neurons.length; i++) {
                    if (Math.random() >= rate) {
                        dropoutMasks[l][i] = true;
                        current.neurons[i].value *= scale;
                    } else {
                        dropoutMasks[l][i] = false;
                        current.neurons[i].value = 0.0f;
                    }
                }
            }
        }

        Layer output = layers[layers.length - 1];
        float[] result = new float[output.neurons.length];

        for (int i = 0; i < result.length; i++) {
            result[i] = output.neurons[i].value;
        }

        return result;
    }

    /**
     * Alias for {@link #forwardPropagate(float[])}.
     *
     * @param inputs one example
     * @return network outputs
     */
    public float[] predict(float[] inputs) {
        return forwardPropagate(inputs);
    }

    /**
     * Sets training mode. When true, dropout is active. When false,
     * dropout is disabled (inference mode).
     *
     * @param training true for training, false for inference
     */
    public void setTraining(boolean training) {
        this.training = training;
    }

    /** Returns true if the network is in training mode. */
    public boolean isTraining() {
        return training;
    }

    /**
     * Sets the dropout rate for a layer.
     *
     * @param layer index of the layer (1..length-1)
     * @param rate  dropout probability in [0, 1)
     */
    public void setDropout(int layer, float rate) {
        if (layer < 1 || layer >= dropoutRates.length) {
            throw new IllegalArgumentException("Invalid layer index");
        }
        if (rate < 0.0f || rate >= 1.0f) {
            throw new IllegalArgumentException("rate must be in [0, 1)");
        }
        dropoutRates[layer] = rate;
    }

    /** Returns the dropout rate for a layer. */
    public float getDropout(int layer) {
        return dropoutRates[layer];
    }

    /**
     * Mean loss over a dataset; forward-only, no updates.
     *
     * @param inputs  examples, [N][inputSize]
     * @param targets desired outputs, [N][outputSize]
     * @return average loss
     * @throws IllegalArgumentException on shape mismatches
     */
    public float calculateLoss(float[][] inputs,
                               float[][] targets) {
        validateDataset(inputs, targets);
        float total = 0.0f;

        for (int i = 0; i < inputs.length; i++) {
            total += lossFunction.loss(
                forwardPropagate(inputs[i]), targets[i]);
        }

        return total / inputs.length;
    }

    /**
     * Trains on one mini-batch: forward pass per sample,
     * backpropagation of output and hidden deltas, gradient
     * accumulation, averaging, then a single optimizer step.
     *
     * @param inputs  batch of examples
     * @param targets batch of targets
     * @throws IllegalArgumentException on shape mismatches
     */
    public void trainBatch(float[][] inputs, float[][] targets) {
        validateDataset(inputs, targets);

        float[] parameters = flattenParameters();
        float[] gradients = new float[parameters.length];

        for (int sample = 0; sample < inputs.length; sample++) {
            float[] prediction =
                forwardPropagate(inputs[sample]);

            if (targets[sample].length != prediction.length) {
                throw new IllegalArgumentException(
                    "Target dimension does not match output");
            }

            float[] outputGradient =
                lossFunction.gradient(
                    prediction, targets[sample]);

            Layer outputLayer = layers[layers.length - 1];

            for (int i = 0;
                 i < outputLayer.neurons.length; i++) {
                Neuron neuron = outputLayer.neurons[i];
                neuron.gradient = outputGradient[i]
                    * activations[layers.length - 1]
                        .derivative(neuron.preActivation, neuron.value);

                // Apply dropout gradient
                if (dropoutRates[layers.length - 1] > 0.0f &&
                    dropoutMasks[layers.length - 1] != null) {
                    if (!dropoutMasks[layers.length - 1][i]) {
                        neuron.gradient = 0.0f;
                    } else {
                        neuron.gradient *=
                            1.0f / (1.0f - dropoutRates[layers.length - 1]);
                    }
                }
            }

            for (int l = layers.length - 2; l > 0; l--) {
                Layer current = layers[l];
                Layer next = layers[l + 1];
                int curLen = current.neurons.length;

                for (int i = 0; i < curLen; i++) {
                    deltaBuffer[i] = 0.0f;
                }

                for (Neuron nextNeuron : next.neurons) {
                    VectorMath.addScaledScalar(
                        deltaBuffer, 0, nextNeuron.gradient,
                        nextNeuron.weights, curLen);
                }

                for (int i = 0; i < curLen; i++) {
                    Neuron neuron = current.neurons[i];
                    neuron.gradient = deltaBuffer[i]
                        * activations[l].derivative(
                            neuron.preActivation, neuron.value);

                    // Apply dropout gradient
                    if (dropoutRates[l] > 0.0f &&
                        dropoutMasks[l] != null) {
                        if (!dropoutMasks[l][i]) {
                            neuron.gradient = 0.0f;
                        } else {
                            neuron.gradient *=
                                1.0f / (1.0f - dropoutRates[l]);
                        }
                    }
                }
            }

            int k = 0;

            for (int l = 1; l < layers.length; l++) {
                Layer previous = layers[l - 1];
                int prevLen = previous.neurons.length;

                for (int j = 0; j < prevLen; j++) {
                    prevBuffer[j] = previous.neurons[j].value;
                }

                for (Neuron neuron : layers[l].neurons) {
                    VectorMath.addScaledScalar(
                        gradients, k, neuron.gradient,
                        prevBuffer, prevLen);
                    k += prevLen;
                    gradients[k++] += neuron.gradient;
                }
            }
        }

        for (int i = 0; i < gradients.length; i++) {
            gradients[i] /= inputs.length;
        }

        optimizer.update(parameters, gradients);
        unflattenParameters(parameters);
    }

    /**
     * Trains the network over the dataset for the configured
     * number of epochs, optionally reshuffling row order each
     * epoch and slicing into mini-batches.
     *
     * @param trainer epochs, batch size, shuffle flag and seed
     * @param inputs  training examples, [N][inputSize]
     * @param targets training targets, [N][outputSize]
     * @throws IllegalArgumentException on shape mismatches
     */
    public void fit(Trainer trainer,
                    float[][] inputs,
                    float[][] targets) {
        validateDataset(inputs, targets);

        int[] order = new int[inputs.length];

        for (int epoch = 0;
             epoch < trainer.getEpochs(); epoch++) {

            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }

            if (trainer.shouldShuffle()) {
                Random random = trainer.getRandom();

                for (int i = order.length - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    int temp = order[i];
                    order[i] = order[j];
                    order[j] = temp;
                }
            }

            int batchSize = trainer.getBatchSize();

            for (int start = 0;
                 start < order.length;
                 start += batchSize) {

                int end =
                    Math.min(start + batchSize, order.length);
                int count = end - start;

                float[][] batchInputs = new float[count][];
                float[][] batchTargets = new float[count][];

                for (int i = 0; i < count; i++) {
                    batchInputs[i] = inputs[order[start + i]];
                    batchTargets[i] = targets[order[start + i]];
                }

                trainBatch(batchInputs, batchTargets);
            }
        }
    }

    private float[] flattenParameters() {
        int count = 0;

        for (int l = 1; l < layers.length; l++) {
            for (Neuron neuron : layers[l].neurons) {
                count += neuron.weights.length + 1;
            }
        }

        float[] result = new float[count];
        int k = 0;

        for (int l = 1; l < layers.length; l++) {
            for (Neuron neuron : layers[l].neurons) {
                for (float weight : neuron.weights) {
                    result[k++] = weight;
                }

                result[k++] = neuron.bias;
            }
        }

        return result;
    }

    /**
     * Returns a copy of all trainable parameters (weights and
     * biases) as a flat array. Useful for inspection, but use
     * {@link #saveModel} for persistence.
     *
     * @return flat parameter vector
     */
    public float[] getParameters() {
        return flattenParameters();
    }

    /** Layers in order, for package-level weight loading ({@link ModelLoader}). */
    Layer[] layerArray() {
        return layers;
    }

    /**
     * Overwrites all trainable parameters from a flat array
     * previously produced by {@link #getParameters} or
     * {@link #loadModel}. Length must match exactly.
     *
     * @param parameters flat parameter vector
     * @throws IllegalArgumentException on length mismatch
     */
    public void setParameters(float[] parameters) {
        if (parameters == null ||
            parameters.length != flattenParameters().length) {
            throw new IllegalArgumentException(
                "Parameter vector length does not match network");
        }

        unflattenParameters(parameters);
    }

    /**
     * Returns the layer sizes of this network.
     *
     * @return topology array, index 0 is the input size
     */
    public int[] getTopology() {
        int[] topo = new int[layers.length];

        for (int i = 0; i < layers.length; i++) {
            topo[i] = layers[i].neurons.length;
        }

        return topo;
    }

    /**
     * Saves the network topology and all parameters to a file.
     * Format is inferred from the extension: .bin, .json, or
     * .csv (default binary).
     *
     * @param file destination path
     * @throws IOException on write failure
     */
    public void saveModel(java.nio.file.Path file)
            throws java.io.IOException {
        saveModel(file, ModelFormat.fromPath(file.toString()));
    }

    /**
     * Saves the network in the explicitly chosen format.
     *
     * @param file   destination path
     * @param format BINARY, JSON, or CSV
     * @throws IOException on write failure
     */
    public void saveModel(java.nio.file.Path file,
                          ModelFormat format)
            throws java.io.IOException {
        switch (format) {
            case JSON:
                saveJson(file);
                break;
            case CSV:
                saveCsv(file);
                break;
            case BINARY:
            default:
                saveBinary(file);
                break;
        }
    }

    /**
     * Loads a network, inferring the format from the file
     * extension. The caller supplies the activations, loss,
     * optimizer, seed and initializer used at training time so
     * the reconstructed network behaves identically for
     * inference.
     *
     * @param file         path produced by saveModel
     * @param activations  one activation per layer
     * @param lossFunction error metric (unused for inference)
     * @param optimizer    update rule (unused for inference)
     * @param seed         RNG seed (unused for inference)
     * @param initializer  weight init (unused for inference)
     * @return a network with the saved weights loaded
     * @throws IOException on read failure or bad format
     */
    public static NeuralNetwork loadModel(
            java.nio.file.Path file,
            ActivationFunction[] activations,
            LossFunction lossFunction,
            Optimizer optimizer,
            long seed,
            WeightInitializer initializer)
            throws java.io.IOException {
        return loadModel(file, ModelFormat.fromPath(
            file.toString()), activations, lossFunction,
            optimizer, seed, initializer);
    }

    /**
     * Loads a network from a file in the explicitly chosen
     * format.
     */
    public static NeuralNetwork loadModel(
            java.nio.file.Path file,
            ModelFormat format,
            ActivationFunction[] activations,
            LossFunction lossFunction,
            Optimizer optimizer,
            long seed,
            WeightInitializer initializer)
            throws java.io.IOException {
        switch (format) {
            case JSON:
                return loadJson(file, activations,
                    lossFunction, optimizer, seed,
                    initializer);
            case CSV:
                return loadCsv(file, activations,
                    lossFunction, optimizer, seed,
                    initializer);
            case BINARY:
            default:
                return loadBinary(file, activations,
                    lossFunction, optimizer, seed,
                    initializer);
        }
    }

    // ── BINARY ──────────────────────────────────────────────

    private void saveBinary(java.nio.file.Path file)
            throws java.io.IOException {
        try (java.io.DataOutputStream out =
                new java.io.DataOutputStream(
                    java.nio.file.Files.newOutputStream(file))) {
            out.writeInt(0x4E4E4C31);            // magic "NNL1"
            out.writeInt(layers.length);          // layer count

            for (Layer layer : layers) {
                out.writeInt(layer.neurons.length);
            }

            for (float w : flattenParameters()) {
                out.writeFloat(w);
            }
        }
    }

    private static NeuralNetwork loadBinary(
            java.nio.file.Path file,
            ActivationFunction[] activations,
            LossFunction lossFunction,
            Optimizer optimizer,
            long seed,
            WeightInitializer initializer)
            throws java.io.IOException {
        try (java.io.DataInputStream in =
                new java.io.DataInputStream(
                    java.nio.file.Files.newInputStream(file))) {
            int magic = in.readInt();

            if (magic != 0x4E4E4C31) {
                throw new java.io.IOException(
                    "Not a valid binary model: " + file);
            }

            int numLayers = in.readInt();
            int[] topology = new int[numLayers];

            for (int i = 0; i < numLayers; i++) {
                topology[i] = in.readInt();
            }

            NeuralNetwork net = new NeuralNetwork(
                topology, activations, lossFunction,
                optimizer, seed, initializer);

            float[] params =
                new float[net.flattenParameters().length];

            for (int i = 0; i < params.length; i++) {
                params[i] = in.readFloat();
            }

            net.unflattenParameters(params);
            return net;
        }
    }

    // ── JSON ────────────────────────────────────────────────

    private void saveJson(java.nio.file.Path file)
            throws java.io.IOException {
        int[] topo = getTopology();
        float[] params = flattenParameters();

        StringBuilder sb = new StringBuilder();
        sb.append("{\n  \"topology\": [");

        for (int i = 0; i < topo.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(topo[i]);
        }

        sb.append("],\n  \"parameters\": [");

        for (int i = 0; i < params.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(Float.toString(params[i]));
        }

        sb.append("]\n}\n");

        java.nio.file.Files.writeString(file, sb.toString());
    }

    private static NeuralNetwork loadJson(
            java.nio.file.Path file,
            ActivationFunction[] activations,
            LossFunction lossFunction,
            Optimizer optimizer,
            long seed,
            WeightInitializer initializer)
            throws java.io.IOException {
        String text =
            java.nio.file.Files.readString(file).trim();

        int[] topology = parseJsonIntArray(text, "topology");
        float[] params =
            parseJsonFloatArray(text, "parameters");

        NeuralNetwork net = new NeuralNetwork(
            topology, activations, lossFunction,
            optimizer, seed, initializer);

        if (params.length != net.flattenParameters().length) {
            throw new java.io.IOException(
                "Parameter count mismatch: file has "
                + params.length + ", network expects "
                + net.flattenParameters().length);
        }

        net.unflattenParameters(params);
        return net;
    }

    // ── CSV ─────────────────────────────────────────────────

    private void saveCsv(java.nio.file.Path file)
            throws java.io.IOException {
        int[] topo = getTopology();
        float[] params = flattenParameters();

        StringBuilder sb = new StringBuilder();
        sb.append("# topology: ");

        for (int i = 0; i < topo.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(topo[i]);
        }

        sb.append('\n');

        for (float w : params) {
            sb.append(Float.toString(w));
            sb.append('\n');
        }

        java.nio.file.Files.writeString(file, sb.toString());
    }

    private static NeuralNetwork loadCsv(
            java.nio.file.Path file,
            ActivationFunction[] activations,
            LossFunction lossFunction,
            Optimizer optimizer,
            long seed,
            WeightInitializer initializer)
            throws java.io.IOException {
        java.util.List<String> lines =
            java.nio.file.Files.readAllLines(file);

        int[] topology = null;
        java.util.List<Float> values =
            new java.util.ArrayList<>();

        for (String line : lines) {
            line = line.trim();

            if (line.isEmpty()) {
                continue;
            }

            if (line.startsWith("# topology:")) {
                String nums = line.substring(
                    "# topology:".length()).trim();
                String[] parts = nums.split(",");
                topology = new int[parts.length];

                for (int i = 0; i < parts.length; i++) {
                    topology[i] =
                        Integer.parseInt(parts[i].trim());
                }
            } else {
                values.add(Float.parseFloat(line));
            }
        }

        if (topology == null) {
            throw new java.io.IOException(
                "CSV missing topology header line");
        }

        NeuralNetwork net = new NeuralNetwork(
            topology, activations, lossFunction,
            optimizer, seed, initializer);

        float[] params = new float[values.size()];
        for (int i = 0; i < params.length; i++) {
            params[i] = values.get(i);
        }

        if (params.length != net.flattenParameters().length) {
            throw new java.io.IOException(
                "Parameter count mismatch in CSV");
        }

        net.unflattenParameters(params);
        return net;
    }

    // ── JSON parsing helpers ────────────────────────────────

    private static int[] parseJsonIntArray(String text,
                                           String key) {
        String marker = "\"" + key + "\"";
        int idx = text.indexOf(marker);

        if (idx < 0) {
            throw new IllegalArgumentException(
                "Missing key: " + key);
        }

        int start = text.indexOf('[', idx);
        int end = text.indexOf(']', start);
        String inner = text.substring(start + 1, end);
        String[] parts = inner.split(",");
        int[] result = new int[parts.length];

        for (int i = 0; i < parts.length; i++) {
            result[i] = Integer.parseInt(parts[i].trim());
        }

        return result;
    }

    private static float[] parseJsonFloatArray(String text,
                                               String key) {
        String marker = "\"" + key + "\"";
        int idx = text.indexOf(marker);

        if (idx < 0) {
            throw new IllegalArgumentException(
                "Missing key: " + key);
        }

        int start = text.indexOf('[', idx);
        int end = text.lastIndexOf(']');
        String inner = text.substring(start + 1, end);
        String[] parts = inner.split(",");
        float[] result = new float[parts.length];

        for (int i = 0; i < parts.length; i++) {
            result[i] = Float.parseFloat(parts[i].trim());
        }

        return result;
    }

    private void unflattenParameters(float[] parameters) {
        int k = 0;

        for (int l = 1; l < layers.length; l++) {
            for (Neuron neuron : layers[l].neurons) {
                for (int j = 0; j < neuron.weights.length; j++) {
                    neuron.weights[j] = parameters[k++];
                }

                neuron.bias = parameters[k++];
            }
        }
    }

    private void validateDataset(float[][] inputs,
                                 float[][] targets) {
        if (inputs == null || targets == null ||
            inputs.length == 0 ||
            inputs.length != targets.length) {
            throw new IllegalArgumentException(
                "Dataset must have matching, nonempty inputs and targets");
        }

        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] == null ||
                inputs[i].length
                    != layers[0].neurons.length ||
                targets[i] == null ||
                targets[i].length
                    != layers[layers.length - 1]
                        .neurons.length) {
                throw new IllegalArgumentException(
                    "Invalid input or target dimensions at row "
                        + i);
            }
        }
    }
}
