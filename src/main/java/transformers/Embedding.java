package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.List;

/**
 * Token embedding layer: looks up a dense vector for each
 * integer token id from a learnable weight matrix of shape
 * {vocabSize, embeddingDim}.
 */
public final class Embedding {

    private final Tensor weight;
    private final int vocabSize;
    private final int embeddingDim;

    /**
     * Creates an embedding layer initialised from N(0, 0.02).
     *
     * @param vocabSize    number of distinct tokens
     * @param embeddingDim size of each embedding vector
     * @param seed         RNG seed for reproducible init
     */
    public Embedding(int vocabSize, int embeddingDim,
                     long seed) {
        if (vocabSize <= 0 || embeddingDim <= 0) {
            throw new IllegalArgumentException(
                "vocabSize and embeddingDim must be positive, got "
                + vocabSize + ", " + embeddingDim);
        }
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.weight = Tensor.paramRandn(0.02f, seed,
            vocabSize, embeddingDim);
    }

    /**
     * Looks up embeddings for a batch of token id sequences.
     *
     * @param ids token ids, length = batchSize * seqLen
     * @param batchSize number of sequences
     * @param seqLen    length of each sequence
     * @return tensor of shape {batchSize, seqLen, embeddingDim}
     */
    public Tensor forward(int[] ids, int batchSize,
                          int seqLen) {
        if (ids.length != batchSize * seqLen) {
            throw new IllegalArgumentException(
                "ids length " + ids.length + " != "
                + batchSize + " * " + seqLen);
        }
        for (int id : ids) {
            if (id < 0 || id >= vocabSize) {
                throw new IllegalArgumentException(
                    "token id " + id + " out of range [0,"
                    + vocabSize + ")");
            }
        }
        return Ops.embedding(weight, ids,
            batchSize, seqLen, embeddingDim);
    }

    /** Returns all trainable parameters of this layer. */
    public List<Tensor> parameters() {
        return List.of(weight);
    }
}
