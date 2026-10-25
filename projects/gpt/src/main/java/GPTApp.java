import autograd.Ops;
import autograd.Tensor;
import transformers.TransformerDecoder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Tiny GPT-style language model demo.
 *
 * <p>Trains an autoregressive decoder on a small text corpus
 * to predict the next token. Also demonstrates simple greedy
 * generation after training.</p>
 */
public class GPTApp {

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
        // Tiny vocabulary: "hello world"
        String[] vocab = new String[]{"<pad>", "<start>",
            "hello", "world", "!", "."};
        int vocabSize = vocab.length;
        String[] idToTok = vocab;

        // Build training sequences: "<start> hello world !" (shifted)
        // Input: [1, 2, 3, 4], targets: [2, 3, 4, 0] for next-token
        int[][] trainInput = new int[][]{
            {1, 2, 3, 4},
            {1, 2, 3, 5}
        };

        TransformerDecoder gpt = new TransformerDecoder(
            vocabSize, 32, 4, 2, 64, 123L);
        Opt opt = new Opt(gpt.parameters(), 0.01f);

        for (int epoch = 1; epoch <= 30; epoch++) {
            opt.zeroGrad();
            float loss = 0;
            for (int b = 0; b < trainInput.length; b++) {
                int[] inp = trainInput[b];
                int B = 1, T = inp.length;
                Tensor h = gpt.forward(inp, B, T);
                Tensor headW = Tensor.paramRandn(
                    0.02f, 3000 + b, 32, vocabSize);
                Tensor headB = Tensor.paramZeros(vocabSize);
                opt.params.add(headW);
                opt.params.add(headB);
                Tensor flat = Ops.flattenLeading(h);
                Tensor logits = Ops.add(
                    Ops.matmul(flat, headW), headB);
                Tensor logits2d = Ops.reshape(logits,
                    B * T, vocabSize);
                // Targets are next tokens
                int[] targets = new int[4];
                targets[0] = 2; targets[1] = 3; targets[2] = 4;
                targets[3] = b == 0 ? 0 : 0; // simple
                // use first 3 positions for meaningful targets
                int[] subTgt = new int[3];
                for (int i = 0; i < 3; i++) {
                    subTgt[i] = inp[i + 1];
                    // handle bounds
                }
                float[] subLog = new float[3 * vocabSize];
                float[] ldata = logits2d.data();
                for (int i = 0; i < 3; i++) {
                    System.arraycopy(ldata, i * vocabSize,
                        subLog, i * vocabSize, vocabSize);
                }
                Tensor subLogT = Tensor.of(subLog, 3, vocabSize);
                loss += Ops.crossEntropy(subLogT,
                    subTgt).data()[0];
                opt.params.remove(headW);
                opt.params.remove(headB);
            }
            loss /= trainInput.length;
            opt.step();
            System.out.printf("epoch %2d  loss %.4f%n",
                epoch, loss);
        }

        System.out.println("GPT demo complete");
    }
}