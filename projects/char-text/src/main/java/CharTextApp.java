import activations.ActivationFunction;
import initialization.WeightInitializer;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.Adam;
import training.Trainer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Project 8 - Character-level text generation.
 *
 * A sliding window of 16 characters is one-hot encoded and fed
 * to a dense network that predicts the next character. After
 * training, new text is produced by sampling from the softmax at
 * different temperatures.
 */
public class CharTextApp {

    private static final String CORPUS =
        "it was the best of times it was the worst of times "
        + "it was the age of wisdom it was the age of "
        + "foolishness it was the epoch of belief it was the "
        + "epoch of incredulity it was the season of light it "
        + "was the season of darkness it was the spring of hope "
        + "it was the winter of despair";

    private static final int WINDOW = 16;
    private static final int EPOCHS = 40;
    private static final long SEED = 42L;

    public static void main(String[] args) {
        char[] alphabet = alphabet(CORPUS);
        int vocab = alphabet.length;
        Map<Character, Integer> index = new LinkedHashMap<>();
        for (int i = 0; i < alphabet.length; i++) {
            index.put(alphabet[i], i);
        }

        System.out.println("Corpus length: " + CORPUS.length()
            + "  vocabulary: " + vocab);

        List<float[]> inputs = new ArrayList<>();
        List<float[]> targets = new ArrayList<>();
        for (int i = 0; i + WINDOW < CORPUS.length(); i++) {
            inputs.add(encode(CORPUS, i, index, vocab));
            targets.add(oneHot(
                index.get(CORPUS.charAt(i + WINDOW)), vocab));
        }

        float[][] xs = inputs.toArray(new float[0][]);
        float[][] ys = targets.toArray(new float[0][]);

        NeuralNetwork net = new NeuralNetwork(
            new int[] {vocab * WINDOW, 128, vocab},
            new ActivationFunction[] {
                ActivationFunction.linear(),
                ActivationFunction.relu(),
                ActivationFunction.linear()
            },
            LossFunction.softmaxCrossEntropy(),
            new Adam(0.001f),
            SEED,
            WeightInitializer.heNormal());

        System.out.println("Training on " + xs.length
            + " (window, next-char) pairs ...");
        net.fit(new Trainer(EPOCHS, 64, true, SEED), xs, ys);
        System.out.printf("Final loss: %.4f%n%n",
            net.calculateLoss(xs, ys));

        String seed = CORPUS.substring(
            CORPUS.length() - WINDOW);
        float[] temperatures = {0.2f, 0.8f, 1.4f};
        for (float temperature : temperatures) {
            System.out.printf("temperature %.1f%n  \"%s\"%n%n",
                temperature,
                sample(net, seed, 60, temperature, index,
                    alphabet, vocab));
        }
    }

    private static String sample(NeuralNetwork net, String seed,
                                 int length, float temperature,
                                 Map<Character, Integer> index,
                                 char[] alphabet, int vocab) {
        StringBuilder text = new StringBuilder(seed);
        float[] context = encode(seed, seed.length() - WINDOW,
            index, vocab);
        Random rand = new Random(7L);

        for (int step = 0; step < length; step++) {
            float[] logits = net.predict(context);
            for (int i = 0; i < logits.length; i++) {
                logits[i] /= temperature;
            }
            float[] probs = softmax(logits);

            int chosen = roulette(probs, rand);
            text.append(alphabet[chosen]);

            shift(context, oneHot(chosen, vocab), vocab);
        }
        return text.toString();
    }

    private static float[] encode(String text, int start,
                                  Map<Character, Integer> index,
                                  int vocab) {
        float[] vector = new float[vocab * WINDOW];
        for (int i = 0; i < WINDOW; i++) {
            Integer id = index.get(text.charAt(start + i));
            if (id != null) {
                vector[i * vocab + id] = 1.0f;
            }
        }
        return vector;
    }

    private static float[] oneHot(int id, int vocab) {
        float[] vector = new float[vocab];
        vector[id] = 1.0f;
        return vector;
    }

    private static void shift(float[] context, float[] next,
                              int vocab) {
        System.arraycopy(context, vocab, context, 0,
            context.length - vocab);
        System.arraycopy(next, 0, context,
            context.length - vocab, vocab);
    }

    private static float[] softmax(float[] logits) {
        float max = logits[0];
        for (float v : logits) {
            if (v > max) {
                max = v;
            }
        }
        float sum = 0.0f;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    private static int roulette(float[] probs, Random rand) {
        float r = rand.nextFloat();
        float cumulative = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r <= cumulative) {
                return i;
            }
        }
        return probs.length - 1;
    }

    private static char[] alphabet(String text) {
        boolean[] seen = new boolean[Character.MAX_VALUE];
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (!seen[c]) {
                seen[c] = true;
                sb.append(c);
            }
        }
        char[] result = new char[sb.length()];
        sb.getChars(0, sb.length(), result, 0);
        return result;
    }
}
