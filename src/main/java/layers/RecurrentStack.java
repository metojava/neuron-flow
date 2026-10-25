package layers;

import optim.Optimizer;

/**
 * Stacked recurrent network: {@code numLayers} layers of RNN, LSTM or GRU cells,
 * optionally bidirectional.
 *
 * <p>Each layer runs one cell per direction. The backward direction processes the
 * sequence in reverse, and its outputs are reversed back and concatenated with the
 * forward outputs, so the output size is {@code hiddenSize * directions}. Layer l > 0
 * receives that concatenated sequence as input.
 *
 * <p>Parameters are the concatenation of each cell's parameters, ordered by layer,
 * then direction.
 */
public final class RecurrentStack {

    /** Cell type. */
    public enum Type { RNN, LSTM, GRU }

    private final Type type;
    private final int inputSize;
    private final int hiddenSize;
    private final int numLayers;
    private final int directions;
    private final boolean gruResetAfter;
    private final RecurrentCell[][] cells; // [layer][direction]

    /**
     * @param type           cell type
     * @param inputSize      size of each input vector
     * @param hiddenSize     hidden size per direction
     * @param numLayers      number of stacked layers, at least 1
     * @param bidirectional  whether to run a backward direction in each layer
     * @param gruResetAfter  for GRU only: use the PyTorch reset-after formulation
     * @param seed           RNG seed
     * @throws IllegalArgumentException on non-positive sizes
     */
    public RecurrentStack(Type type, int inputSize, int hiddenSize, int numLayers,
                          boolean bidirectional, boolean gruResetAfter, long seed) {
        if (type == null || inputSize <= 0 || hiddenSize <= 0 || numLayers <= 0) {
            throw new IllegalArgumentException("Invalid recurrent stack configuration");
        }
        this.type = type;
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.numLayers = numLayers;
        this.directions = bidirectional ? 2 : 1;
        this.gruResetAfter = gruResetAfter;
        this.cells = new RecurrentCell[numLayers][directions];
        for (int l = 0; l < numLayers; l++) {
            int in = l == 0 ? inputSize : hiddenSize * directions;
            for (int d = 0; d < directions; d++) {
                cells[l][d] = create(type, in, hiddenSize, gruResetAfter, seed + 2L * l + d);
            }
        }
    }

    private static RecurrentCell create(Type type, int in, int h, boolean resetAfter, long seed) {
        switch (type) {
            case RNN: return new RNN(in, h, seed);
            case LSTM: return new LSTM(in, h, seed);
            default: return new GRU(in, h, resetAfter, seed);
        }
    }

    /** @return output size per time step: hiddenSize times the number of directions */
    public int outputSize() {
        return hiddenSize * directions;
    }

    public Type type() {
        return type;
    }

    public int inputSize() {
        return inputSize;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public int numLayers() {
        return numLayers;
    }

    public int directions() {
        return directions;
    }

    public boolean isGruResetAfter() {
        return gruResetAfter;
    }

    /** Cell for a layer and direction; package access for {@link ModelLoader}. */
    RecurrentCell cell(int layer, int direction) {
        return cells[layer][direction];
    }

    /**
     * @param sequence [seqLen][inputSize]
     * @return [seqLen][outputSize()]
     */
    public float[][] forward(float[][] sequence) {
        float[][] x = sequence;
        for (int l = 0; l < numLayers; l++) {
            float[][] fwd = cells[l][0].forward(x);
            if (directions == 1) {
                x = fwd;
            } else {
                float[][] bwd = reverse(cells[l][1].forward(reverse(x)));
                x = concatColumns(fwd, bwd);
            }
        }
        return x;
    }

    /**
     * Backpropagation through the last {@link #forward} call. Accumulates gradients in
     * every cell.
     *
     * @param dOutput [seqLen][outputSize()]
     * @return [seqLen][inputSize]
     */
    public float[][] backward(float[][] dOutput) {
        float[][] d = dOutput;
        for (int l = numLayers - 1; l >= 0; l--) {
            float[][] dIn = cells[l][0].backward(sliceColumns(d, 0, hiddenSize));
            if (directions == 2) {
                float[][] dBwd = cells[l][1].backward(
                        reverse(sliceColumns(d, hiddenSize, 2 * hiddenSize)));
                dIn = addMatrices(dIn, reverse(dBwd));
            }
            d = dIn;
        }
        return d;
    }

    /** @return total number of trainable parameters */
    public int parameterCount() {
        int total = 0;
        for (RecurrentCell[] layer : cells) {
            for (RecurrentCell c : layer) total += c.parameterCount();
        }
        return total;
    }

    /** Copies all parameters, ordered by layer then direction. */
    public void copyParametersTo(float[] buf, int off) {
        int p = off;
        for (RecurrentCell[] layer : cells) {
            for (RecurrentCell c : layer) {
                c.copyParametersTo(buf, p);
                p += c.parameterCount();
            }
        }
    }

    /** Inverse of {@link #copyParametersTo}. */
    public void copyParametersFrom(float[] buf, int off) {
        int p = off;
        for (RecurrentCell[] layer : cells) {
            for (RecurrentCell c : layer) {
                c.copyParametersFrom(buf, p);
                p += c.parameterCount();
            }
        }
    }

    /** Copies accumulated gradients in the same layout as {@link #copyParametersTo}. */
    public void copyGradientsTo(float[] buf, int off) {
        int p = off;
        for (RecurrentCell[] layer : cells) {
            for (RecurrentCell c : layer) {
                c.copyGradientsTo(buf, p);
                p += c.parameterCount();
            }
        }
    }

    /** Clears accumulated gradients in every cell. */
    public void zeroGrad() {
        for (RecurrentCell[] layer : cells) {
            for (RecurrentCell c : layer) c.zeroGrad();
        }
    }

    /** Applies accumulated gradients with SGD. */
    public void update(float learningRate, int batchSize) {
        for (RecurrentCell[] layer : cells) {
            for (RecurrentCell c : layer) c.update(learningRate, batchSize);
        }
    }

    /**
     * Applies one optimizer step to all parameters, using gradients averaged over the batch.
     *
     * @param optimizer update rule (for example {@link optim.Adam})
     * @param batchSize number of sequences summed into the gradients
     */
    public void update(Optimizer optimizer, int batchSize) {
        float[] p = new float[parameterCount()];
        float[] g = new float[p.length];
        copyParametersTo(p, 0);
        copyGradientsTo(g, 0);
        for (int i = 0; i < g.length; i++) g[i] /= batchSize;
        optimizer.update(p, g);
        copyParametersFrom(p, 0);
    }

    // ---- matrix helpers ----

    static float[][] reverse(float[][] m) {
        float[][] out = new float[m.length][];
        for (int t = 0; t < m.length; t++) {
            out[t] = m[m.length - 1 - t].clone();
        }
        return out;
    }

    private static float[][] sliceColumns(float[][] m, int from, int to) {
        float[][] out = new float[m.length][to - from];
        for (int t = 0; t < m.length; t++) {
            System.arraycopy(m[t], from, out[t], 0, to - from);
        }
        return out;
    }

    private static float[][] concatColumns(float[][] a, float[][] b) {
        float[][] out = new float[a.length][a[0].length + b[0].length];
        for (int t = 0; t < a.length; t++) {
            System.arraycopy(a[t], 0, out[t], 0, a[t].length);
            System.arraycopy(b[t], 0, out[t], a[t].length, b[t].length);
        }
        return out;
    }

    private static float[][] addMatrices(float[][] a, float[][] b) {
        float[][] out = new float[a.length][a[0].length];
        for (int t = 0; t < a.length; t++) {
            for (int j = 0; j < a[t].length; j++) out[t][j] = a[t][j] + b[t][j];
        }
        return out;
    }
}
