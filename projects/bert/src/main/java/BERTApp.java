import autograd.Ops;
import autograd.Tensor;
import transformers.TransformerEncoder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Tiny BERT-style masked language model demo.
 *
 * <p>We train on a small artificial corpus: the task is to
 * predict masked tokens. This demonstrates the autograd engine,
 * transformer encoder, and the training loop with simple SGD.</p>
 */
public class BERTApp {

    private static class Opt {
        private final List<Tensor> params;
        private final float lr;

        Opt(List<Tensor> params, float lr) {
            this.params = params;
            this.lr = lr;
        }

        void step() {
            for (Tensor p : params) {
                if (!p.hasGrad()) continue;
                float[] pg = p.grad();
                float[] pd = p.data();
                for (int i = 0; i < pg.length; i++) {
                    pd[i] -= lr * pg[i];
                }
            }
        }

        void zeroGrad() {
            for (Tensor p : params) {
                p.zeroGrad();
            }
        }
    }

    public static void main(String[] args) {
        // Small toy vocabulary
        String[] vocab = new String[]{"[PAD]", "[MASK]", "the",
            "cat", "sat", "on", "mat", "dog", "ran", "in",
            "room", "and"};
        int vocabSize = vocab.length;

        // Training data: (masked sequence, target for masked positions)
        // Example: "the cat sat on mat" masked -> predict cat/sat
        int[][] batchIds = new int[][]{
            // [the, MASK, MASK, on, mat]
            {2, 1, 1, 5, 6},
            // [dog, MASK, in, MASK, room]
            {7, 1, 9, 1, 10}
        };
        // Targets for masked positions (row-major aligned)
        int[] targets = new int[]{3, 4, 8, 10};

        TransformerEncoder bert = new TransformerEncoder(
            vocabSize, 32, 4, 2, 64, 123L);
        Opt opt = new Opt(bert.parameters(), 0.01f);

        Random rand = new Random(123);
        for (int epoch = 1; epoch <= 20; epoch++) {
            opt.zeroGrad();
            float loss = 0;
            for (int b = 0; b < batchIds.length; b++) {
                int[] ids = batchIds[b];
                int B = 1, T = ids.length;
                Tensor enc = bert.forward(ids, B, T);
                // Build head
                Tensor headW = Tensor.paramRandn(
                    0.02f, 2000 + b, 32, vocabSize);
                Tensor headB = Tensor.paramZeros(vocabSize);
                opt.params.add(headW);
                opt.params.add(headB);
                Tensor flat = Ops.flattenLeading(enc);
                Tensor flatLogits = Ops.add(
                    Ops.matmul(flat, headW), headB);
                Tensor logits2d = Ops.reshape(flatLogits,
                    B * T, vocabSize);
                int[] tgt = new int[]{
                    targets[b * 2], targets[b * 2 + 1]
                };
                // Loss only for first 2 masked positions per row
                loss += crossEntropyFor(logits2d, B, T, tgt, vocabSize).data()[0];
                opt.params.remove(headW);
                opt.params.remove(headB);
            }
            loss /= batchIds.length;
            opt.step();
            System.out.printf("epoch %2d  loss %.4f%n",
                epoch, loss);
        }

        System.out.println("BERT demo complete");
    }

    // Simplified cross-entropy over masked positions
    private static Tensor crossEntropyFor(
            Tensor logits2d, int B, int T,
            int[] tgt, int vocabSize) {
        // Use first 2 positions only (masked)
        int[] smallTgt = new int[]{tgt[0], tgt[1]};
        // Take logits for positions 1,2
        float[] sub = new float[2 * vocabSize];
        float[] ldata = logits2d.data();
        System.arraycopy(ldata, 1 * vocabSize, sub, 0,
            vocabSize);
        System.arraycopy(ldata, 2 * vocabSize, sub,
            vocabSize, vocabSize);
        Tensor subLog = Tensor.of(sub, 2, vocabSize);
        return Ops.crossEntropy(subLog, smallTgt);
    }
}