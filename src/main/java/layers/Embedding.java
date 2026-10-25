package layers;

import java.util.Random;

/**
 * Embedding layer: maps integer token indices to dense vectors.
 * Weight matrix is [vocabSize][embeddingDim]. Forward looks up
 * the row for each token; backward accumulates gradients per token.
 */
public final class Embedding {

    private final int vocabSize;
    private final int embeddingDim;
    private final float[][] weights;
    private final float[][] weightGrads;
    private int[] lastInput;

    /**
     * @param vocabSize    number of distinct tokens
     * @param embeddingDim dimension of each embedding vector
     * @param seed         RNG seed for initialization
     */
    public Embedding(int vocabSize, int embeddingDim, long seed) {
        if (vocabSize <= 0 || embeddingDim <= 0) {
            throw new IllegalArgumentException("vocabSize and embeddingDim must be positive");
        }
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.weights = new float[vocabSize][embeddingDim];
        this.weightGrads = new float[vocabSize][embeddingDim];

        Random rand = new Random(seed);
        float scale = (float) (1.0 / Math.sqrt(embeddingDim));
        for (int v = 0; v < vocabSize; v++) {
            for (int d = 0; d < embeddingDim; d++) {
                weights[v][d] = (float) (rand.nextGaussian() * scale);
            }
        }
    }

    /**
     * Forward pass: looks up embeddings for a sequence of token indices.
     *
     * @param tokens array of token indices
     * @return [seqLen][embeddingDim] matrix
     */
    public float[][] forward(int[] tokens) {
        if (tokens == null) throw new IllegalArgumentException("tokens must not be null");
        lastInput = tokens.clone();
        float[][] out = new float[tokens.length][embeddingDim];
        for (int t = 0; t < tokens.length; t++) {
            int idx = tokens[t];
            if (idx < 0 || idx >= vocabSize) {
                throw new IllegalArgumentException("token index out of range: " + idx);
            }
            System.arraycopy(weights[idx], 0, out[t], 0, embeddingDim);
        }
        return out;
    }

    /**
     * Backward pass: accumulates gradients into the embedding weights.
     *
     * @param grad gradient w.r.t. output, [seqLen][embeddingDim]
     * @return null (no gradient w.r.t. input indices)
     */
    public void backward(float[][] grad) {
        if (lastInput == null) {
            throw new IllegalStateException("Call forward before backward");
        }
        if (grad.length != lastInput.length) {
            throw new IllegalArgumentException("Gradient length mismatch");
        }
        for (int t = 0; t < lastInput.length; t++) {
            int idx = lastInput[t];
            for (int d = 0; d < embeddingDim; d++) {
                weightGrads[idx][d] += grad[t][d];
            }
        }
    }

    /** Applies accumulated gradients with SGD. */
    public void update(float learningRate, int batchSize) {
        float scale = learningRate / batchSize;
        for (int v = 0; v < vocabSize; v++) {
            for (int d = 0; d < embeddingDim; d++) {
                weights[v][d] -= scale * weightGrads[v][d];
                weightGrads[v][d] = 0.0f;
            }
        }
    }

    /** Clears accumulated gradients. */
    public void zeroGrad() {
        for (int v = 0; v < vocabSize; v++) {
            java.util.Arrays.fill(weightGrads[v], 0.0f);
        }
    }

    /**
     * Replaces the whole embedding table, for example with weights loaded
     * from a model file. The table is validated before any value is copied.
     *
     * @param table [vocabSize][embeddingDim] values
     * @throws IllegalArgumentException if the shape does not match
     */
    public void loadWeights(float[][] table) {
        if (table.length != vocabSize) {
            throw new IllegalArgumentException(
                "Expected " + vocabSize + " rows, got " + table.length);
        }
        for (float[] row : table) {
            if (row.length != embeddingDim) {
                throw new IllegalArgumentException(
                    "Expected rows of " + embeddingDim + " values, got " + row.length);
            }
        }
        for (int v = 0; v < vocabSize; v++) {
            System.arraycopy(table[v], 0, weights[v], 0, embeddingDim);
        }
    }

    /** Returns the embedding weight for a token. */
    public float[] weight(int token) {
        return weights[token].clone();
    }

    public int vocabSize() {
        return vocabSize;
    }

    public int embeddingDim() {
        return embeddingDim;
    }

    public int parameterCount() {
        return vocabSize * embeddingDim;
    }
}
