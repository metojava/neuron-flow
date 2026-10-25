package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.List;
import java.util.Random;

/**
 * Combined token, position, and (optional) segment embeddings.
 * Used by BERT-style models.
 */
public final class PositionalEmbedding {

    private final Embedding tokenEmb;
    private final Embedding positionEmb;
    private final Embedding segmentEmb;
    private final boolean useSegments;
    private final int dModel;
    private final int maxSeqLen;

    /**
     * Creates positional embeddings with optional segment embeddings.
     *
     * @param vocabSize    vocabulary size
     * @param dModel       model dimension
     * @param maxSeqLen    maximum sequence length
     * @param typeVocabSize segment vocabulary size (2 for BERT)
     * @param useSegments  whether to include segment embeddings
     * @param seed         RNG seed
     */
    public PositionalEmbedding(int vocabSize, int dModel, int maxSeqLen,
                               int typeVocabSize, boolean useSegments, long seed) {
        if (vocabSize <= 0 || dModel <= 0 || maxSeqLen <= 0) {
            throw new IllegalArgumentException("dimensions must be positive");
        }
        this.dModel = dModel;
        this.maxSeqLen = maxSeqLen;
        this.useSegments = useSegments;

        this.tokenEmb = new Embedding(vocabSize, dModel, seed);
        this.positionEmb = new Embedding(maxSeqLen, dModel, seed + 1);
        this.segmentEmb = useSegments ? new Embedding(typeVocabSize, dModel, seed + 2) : null;
    }

    /**
     * Computes combined embeddings.
     *
     * @param inputIds      token ids, length = batchSize * seqLen
     * @param segmentIds    segment ids (same length), or null if not used
     * @param batchSize     batch size
     * @param seqLen        sequence length
     * @return embeddings of shape {batchSize, seqLen, dModel}
     */
    public Tensor forward(int[] inputIds, int[] segmentIds,
                          int batchSize, int seqLen) {
        if (inputIds.length != batchSize * seqLen) {
            throw new IllegalArgumentException("inputIds length mismatch");
        }
        if (useSegments) {
            if (segmentIds == null || segmentIds.length != batchSize * seqLen) {
                throw new IllegalArgumentException("segmentIds required");
            }
        }

        Tensor tokenEmbs = tokenEmb.forward(inputIds, batchSize, seqLen);
        Tensor posEmbs = positionEmb.forward(createPositions(batchSize, seqLen),
                                              batchSize, seqLen);

        Tensor x = Ops.add(tokenEmbs, posEmbs);

        if (useSegments && segmentEmb != null) {
            Tensor segEmbs = segmentEmb.forward(segmentIds, batchSize, seqLen);
            x = Ops.add(x, segEmbs);
        }

        return x;
    }

    /**
     * Computes combined embeddings from pre-computed token embeddings.
     *
     * @param tokenEmbs    pre-computed token embeddings of shape {batchSize, seqLen, dModel}
     * @param segmentIds   segment ids (same length), or null if not used
     * @param batchSize    batch size
     * @param seqLen       sequence length
     * @return embeddings of shape {batchSize, seqLen, dModel}
     */
    public Tensor forward(Tensor tokenEmbs, int[] segmentIds,
                          int batchSize, int seqLen) {
        Tensor posEmbs = positionEmb.forward(createPositions(batchSize, seqLen),
                                              batchSize, seqLen);

        Tensor x = Ops.add(tokenEmbs, posEmbs);

        if (useSegments && segmentEmb != null) {
            if (segmentIds == null || segmentIds.length != batchSize * seqLen) {
                throw new IllegalArgumentException("segmentIds required");
            }
            Tensor segEmbs = segmentEmb.forward(segmentIds, batchSize, seqLen);
            x = Ops.add(x, segEmbs);
        }

        return x;
    }

    private int[] createPositions(int batchSize, int seqLen) {
        int[] pos = new int[batchSize * seqLen];
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < seqLen; i++) {
                pos[b * seqLen + i] = i;
            }
        }
        return pos;
    }

    public List<Tensor> parameters() {
        List<Tensor> p = new java.util.ArrayList<>();
        p.addAll(tokenEmb.parameters());
        p.addAll(positionEmb.parameters());
        if (segmentEmb != null) p.addAll(segmentEmb.parameters());
        return p;
    }
}