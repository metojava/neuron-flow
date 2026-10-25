package transformers;

import java.util.Random;

/**
 * MLM masking utilities for BERT pretraining.
 * Implements the standard BERT masking strategy:
 * - 15% of tokens are selected for masking
 * - Of those: 80% replaced with [MASK], 10% random, 10% unchanged
 */
public final class MLMMasker {

    private final int maskTokenId;
    private final int vocabSize;
    private final Random random;

    public MLMMasker(int maskTokenId, int vocabSize, long seed) {
        this.maskTokenId = maskTokenId;
        this.vocabSize = vocabSize;
        this.random = new Random(seed);
    }

    /**
     * Creates masked input and labels for MLM.
     *
     * @param inputIds input token ids [B, T]
     * @return MaskedResult with masked input_ids, labels (-100 for non-masked), and mask positions
     */
    public MaskedResult mask(int[] inputIds) {
        int B = 1; // assuming single sequence for now
        int T = inputIds.length;

        int[] maskedIds = inputIds.clone();
        int[] labels = new int[inputIds.length];
        int[] maskPositions = new int[inputIds.length];
        int maskCount = 0;

        for (int i = 0; i < inputIds.length; i++) {
            // Don't mask special tokens (CLS, SEP, PAD) - simplified
            int tokenId = inputIds[i];
            if (isSpecialToken(inputIds[i])) {
                labels[i] = -100;
                continue;
            }

            if (random.nextDouble() < 0.15) { // 15% masking
                labels[i] = inputIds[i]; // keep original for loss
                maskPositions[maskCount++] = i;

                double rand = random.nextDouble();
                if (rand < 0.8) {
                    // 80% replace with [MASK]
                    maskedIds[i] = 103; // [MASK] token ID
                } else if (rand < 0.9) {
                    // 10% random token
                    maskedIds[i] = random.nextInt(vocabSize);
                } else {
                    // 10% keep original
                    // maskedIds[i] stays the same
                }
            } else {
                labels[i] = -100; // ignore in loss
            }
        }

        // Trim mask positions
        int[] finalPositions = new int[maskCount];
        System.arraycopy(maskPositions, 0, maskPositions, 0, maskCount);

        return new MaskedResult(maskedIds, labels, maskCount);
    }

    private boolean isSpecialToken(int tokenId) {
        // Simplified: treat CLS(101), SEP(102), PAD(0) as special
        return tokenId == 0 || tokenId == 101 || tokenId == 102 || tokenId == 103;
    }

    public static final class MaskedResult {
        public final int[] inputIds;
        public final int[] labels;
        public final int maskCount;

        public MaskedResult(int[] inputIds, int[] labels, int maskCount) {
            this.inputIds = inputIds;
            this.labels = labels;
            this.maskCount = maskCount;
        }
    }
}