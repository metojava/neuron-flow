package layers;

import data.model.Tensor;
import data.model.WeightFile;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Whole-model loading and export for the built-in classifiers, using named tensors.
 *
 * <p>Tensor names:
 * <ul>
 *   <li>{@link DenseClassifier}: {@code layers.<i>.weight} and {@code layers.<i>.bias}
 *       for each fully connected layer i, matching PyTorch's nn.Sequential naming.</li>
 *   <li>{@link ConvClassifier}: {@code conv1}, {@code conv2}, {@code dense1} and
 *       {@code dense2}, each with {@code .weight} and {@code .bias} suffixes.</li>
 * </ul>
 * Weights use PyTorch layouts: Linear [out, in], Conv2d [out, in, kH, kW].
 *
 * <p>Loading is all-or-nothing: if any tensor is missing or has the wrong shape,
 * the model's previous parameters are restored and an exception is thrown.
 */
public final class ModelLoader {

    private ModelLoader() {
    }

    // ---- DenseClassifier ----

    /**
     * Exports the classifier's weights under the names listed in the class documentation.
     *
     * @param model the model to export
     * @return a weight file holding every layer's weights and biases
     */
    public static WeightFile exportDenseClassifier(DenseClassifier model) {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        Dense[] layers = model.denseLayers();
        for (int l = 0; l < layers.length; l++) {
            addDense(tensors, "layers." + l, layers[l]);
        }
        return new WeightFile(new LinkedHashMap<>(), tensors);
    }

    /**
     * Loads weights into an existing classifier with the same layer sizes.
     *
     * @param model destination model
     * @param file  weight file with the names from {@link #exportDenseClassifier}
     * @throws IllegalArgumentException if a tensor is missing or has the wrong shape;
     *                                  the model is left unchanged
     */
    public static void loadDenseClassifier(DenseClassifier model, WeightFile file) {
        float[] snapshot = model.getParameters().clone();
        try {
            Dense[] layers = model.denseLayers();
            for (int l = 0; l < layers.length; l++) {
                loadDense(file.tensors(), "layers." + l, layers[l]);
            }
        } catch (RuntimeException e) {
            model.setParameters(snapshot);
            throw e;
        }
    }

    // ---- ConvClassifier ----

    /**
     * Exports the classifier's weights under the names listed in the class documentation.
     *
     * @param model the model to export
     * @return a weight file holding conv1, conv2, dense1 and dense2
     */
    public static WeightFile exportConvClassifier(ConvClassifier model) {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        addConv(tensors, "conv1", model.conv1());
        addConv(tensors, "conv2", model.conv2());
        addDense(tensors, "dense1", model.dense1());
        addDense(tensors, "dense2", model.dense2());
        return new WeightFile(new LinkedHashMap<>(), tensors);
    }

    /**
     * Loads weights into an existing classifier with the same architecture.
     *
     * @param model destination model
     * @param file  weight file with the names from {@link #exportConvClassifier}
     * @throws IllegalArgumentException if a tensor is missing or has the wrong shape;
     *                                  the model is left unchanged
     */
    public static void loadConvClassifier(ConvClassifier model, WeightFile file) {
        float[] snapshot = model.getParameters().clone();
        try {
            loadConv(file.tensors(), "conv1", model.conv1());
            loadConv(file.tensors(), "conv2", model.conv2());
            loadDense(file.tensors(), "dense1", model.dense1());
            loadDense(file.tensors(), "dense2", model.dense2());
        } catch (RuntimeException e) {
            model.setParameters(snapshot);
            throw e;
        }
    }

    // ---- NeuralNetwork (fully connected, built from Layer/Neuron) ----

    /**
     * Exports a {@link NeuralNetwork}: for each layer after the input, tensors
     * {@code layers.<i>.weight} [out, in] and {@code layers.<i>.bias} [out],
     * where row j of the weight is neuron j's incoming weights.
     *
     * @param net the network to export
     * @return a weight file with one weight and one bias tensor per layer
     */
    public static WeightFile exportNeuralNetwork(NeuralNetwork net) {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        Layer[] layers = net.layerArray();
        for (int l = 1; l < layers.length; l++) {
            Neuron[] neurons = layers[l].neurons;
            int out = neurons.length;
            int in = layers[l - 1].neurons.length;
            String prefix = "layers." + (l - 1);
            float[] w = new float[out * in];
            float[] b = new float[out];
            for (int j = 0; j < out; j++) {
                System.arraycopy(neurons[j].weights, 0, w, j * in, in);
                b[j] = neurons[j].bias;
            }
            tensors.put(prefix + ".weight", new Tensor(prefix + ".weight", new int[] {out, in}, w));
            tensors.put(prefix + ".bias", new Tensor(prefix + ".bias", new int[] {out}, b));
        }
        return new WeightFile(new LinkedHashMap<>(), tensors);
    }

    /**
     * Loads weights into a network with the same topology.
     *
     * @param net  destination network
     * @param file weight file with the names from {@link #exportNeuralNetwork}
     * @throws IllegalArgumentException if a tensor is missing or has the wrong shape;
     *                                  the network is left unchanged
     */
    public static void loadNeuralNetwork(NeuralNetwork net, WeightFile file) {
        float[] snapshot = net.getParameters().clone();
        try {
            Layer[] layers = net.layerArray();
            for (int l = 1; l < layers.length; l++) {
                Neuron[] neurons = layers[l].neurons;
                int out = neurons.length;
                int in = layers[l - 1].neurons.length;
                String prefix = "layers." + (l - 1);
                Tensor w = require(file.tensors(), prefix + ".weight");
                Tensor b = require(file.tensors(), prefix + ".bias");
                requireShape(w, out, in);
                requireShape(b, out);
                for (int j = 0; j < out; j++) {
                    System.arraycopy(w.data(), j * in, neurons[j].weights, 0, in);
                    neurons[j].bias = b.data()[j];
                }
            }
        } catch (RuntimeException e) {
            net.setParameters(snapshot);
            throw e;
        }
    }

    private static void requireShape(Tensor t, int... expected) {
        int[] s = t.shape();
        boolean ok = s.length == expected.length;
        for (int i = 0; ok && i < s.length; i++) {
            ok = s[i] == expected[i];
        }
        if (!ok) {
            throw new IllegalArgumentException("Tensor " + t.name() + ": expected shape "
                    + java.util.Arrays.toString(expected) + ", got " + java.util.Arrays.toString(s));
        }
    }

    // ---- RecurrentStack (PyTorch nn.RNN / nn.LSTM / nn.GRU naming) ----

    /**
     * Exports a recurrent stack with PyTorch names: {@code weight_ih_l<k>},
     * {@code weight_hh_l<k>}, {@code bias_ih_l<k>}, {@code bias_hh_l<k>}, with the suffix
     * {@code _reverse} for the backward direction.
     *
     * @param stack the stack to export
     * @return a weight file in PyTorch layout
     * @throws IllegalArgumentException for a reset-before GRU, which PyTorch cannot represent
     */
    public static WeightFile exportRecurrentStack(RecurrentStack stack) {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        for (int l = 0; l < stack.numLayers(); l++) {
            for (int d = 0; d < stack.directions(); d++) {
                String sfx = "_l" + l + (d == 1 ? "_reverse" : "");
                exportCell(tensors, sfx, stack, stack.cell(l, d));
            }
        }
        return new WeightFile(new LinkedHashMap<>(), tensors);
    }

    /**
     * Loads PyTorch-named weights into a stack with the same type, sizes and layer count.
     *
     * @param stack destination stack
     * @param file  weight file with names from {@link #exportRecurrentStack}
     * @throws IllegalArgumentException on missing tensors or shape mismatch; the stack is
     *                                  left unchanged
     */
    public static void loadRecurrentStack(RecurrentStack stack, WeightFile file) {
        float[] snapshot = new float[stack.parameterCount()];
        stack.copyParametersTo(snapshot, 0);
        try {
            for (int l = 0; l < stack.numLayers(); l++) {
                for (int d = 0; d < stack.directions(); d++) {
                    String sfx = "_l" + l + (d == 1 ? "_reverse" : "");
                    loadCell(file.tensors(), sfx, stack, stack.cell(l, d));
                }
            }
        } catch (RuntimeException e) {
            stack.copyParametersFrom(snapshot, 0);
            throw e;
        }
    }

    private static int gatesOf(RecurrentStack.Type type) {
        switch (type) {
            case RNN: return 1;
            case LSTM: return 4;
            default: return 3;
        }
    }

    private static void loadCell(Map<String, Tensor> in, String sfx, RecurrentStack stack,
                                 RecurrentCell cell) {
        Tensor wih = require(in, "weight_ih" + sfx);
        Tensor whh = require(in, "weight_hh" + sfx);
        Tensor bih = require(in, "bias_ih" + sfx);
        Tensor bhh = require(in, "bias_hh" + sfx);
        switch (stack.type()) {
            case RNN:
                WeightImporter.loadRnn((RNN) cell, wih, whh, bih, bhh);
                break;
            case LSTM:
                WeightImporter.loadLstm((LSTM) cell, wih, whh, bih, bhh);
                break;
            default:
                WeightImporter.loadGru((GRU) cell, wih, whh, bih, bhh);
                break;
        }
    }

    private static void exportCell(Map<String, Tensor> out, String sfx, RecurrentStack stack,
                                   RecurrentCell cell) {
        int h = cell.hiddenSize();
        int in = cell.inputSize();
        int g = gatesOf(stack.type());
        int rows = g * h;
        int rowLen = in + h;
        float[] flat = new float[cell.parameterCount()];
        cell.copyParametersTo(flat, 0);

        float[] wih = new float[rows * in];
        float[] whh = new float[rows * h];
        float[] bih = new float[rows];
        float[] bhh = new float[rows];
        int bBase = rows * rowLen;

        if (stack.type() == RecurrentStack.Type.RNN) {
            System.arraycopy(flat, 0, wih, 0, rows * in);
            System.arraycopy(flat, rows * in, whh, 0, rows * h);
            System.arraycopy(flat, bBase, bih, 0, rows);
        } else if (stack.type() == RecurrentStack.Type.LSTM) {
            int[] perm = {1, 0, 2, 3}; // PyTorch block k <- our block perm[k]
            for (int r = 0; r < rows; r++) {
                int src = perm[r / h] * h + r % h;
                System.arraycopy(flat, src * rowLen, wih, r * in, in);
                System.arraycopy(flat, src * rowLen + in, whh, r * h, h);
                bih[r] = flat[bBase + src];
            }
        } else {
            if (!stack.isGruResetAfter()) {
                throw new IllegalArgumentException(
                    "PyTorch export requires a reset-after GRU");
            }
            int bhnBase = bBase + rows;
            for (int r = 0; r < rows; r++) {
                float sign = (r / h == 1) ? -1f : 1f;
                for (int j = 0; j < in; j++) wih[r * in + j] = sign * flat[r * rowLen + j];
                for (int j = 0; j < h; j++) whh[r * h + j] = sign * flat[r * rowLen + in + j];
                if (r < 2 * h) {
                    bih[r] = sign * flat[bBase + r];
                    bhh[r] = 0f;
                } else {
                    bih[r] = flat[bBase + r];
                    bhh[r] = flat[bhnBase + (r - 2 * h)];
                }
            }
        }
        out.put("weight_ih" + sfx, new Tensor("weight_ih" + sfx, new int[] {rows, in}, wih));
        out.put("weight_hh" + sfx, new Tensor("weight_hh" + sfx, new int[] {rows, h}, whh));
        out.put("bias_ih" + sfx, new Tensor("bias_ih" + sfx, new int[] {rows}, bih));
        out.put("bias_hh" + sfx, new Tensor("bias_hh" + sfx, new int[] {rows}, bhh));
    }

    // ---- helpers ----

    private static void addDense(Map<String, Tensor> out, String prefix, Dense d) {
        int in = d.in;
        int outDim = d.out;
        float[] flat = new float[d.parameterCount()];
        d.copyParametersTo(flat, 0);
        // Dense stores [in][out]; export as PyTorch [out][in].
        float[] w = new float[outDim * in];
        for (int j = 0; j < outDim; j++) {
            for (int i = 0; i < in; i++) {
                w[j * in + i] = flat[i * outDim + j];
            }
        }
        float[] b = new float[outDim];
        System.arraycopy(flat, in * outDim, b, 0, outDim);
        out.put(prefix + ".weight", new Tensor(prefix + ".weight", new int[] {outDim, in}, w));
        out.put(prefix + ".bias", new Tensor(prefix + ".bias", new int[] {outDim}, b));
    }

    private static void loadDense(Map<String, Tensor> in, String prefix, Dense d) {
        WeightImporter.loadDense(d, require(in, prefix + ".weight"), require(in, prefix + ".bias"));
    }

    private static void addConv(Map<String, Tensor> out, String prefix, Conv2D c) {
        float[][][][] w = c.weights;
        int oc = w.length;
        int ic = w[0].length;
        int kh = w[0][0].length;
        int kw = w[0][0][0].length;
        float[] flat = new float[oc * ic * kh * kw];
        int k = 0;
        for (float[][][] filter : w) {
            for (float[][] channel : filter) {
                for (float[] row : channel) {
                    System.arraycopy(row, 0, flat, k, kw);
                    k += kw;
                }
            }
        }
        out.put(prefix + ".weight", new Tensor(prefix + ".weight", new int[] {oc, ic, kh, kw}, flat));
        out.put(prefix + ".bias", new Tensor(prefix + ".bias", new int[] {oc}, c.biases.clone()));
    }

    private static void loadConv(Map<String, Tensor> in, String prefix, Conv2D c) {
        WeightImporter.loadConv2D(c, require(in, prefix + ".weight"), require(in, prefix + ".bias"));
    }

    private static Tensor require(Map<String, Tensor> tensors, String name) {
        Tensor t = tensors.get(name);
        if (t == null) {
            throw new IllegalArgumentException("Missing tensor: " + name);
        }
        return t;
    }
}
