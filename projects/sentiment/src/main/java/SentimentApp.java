import activations.ActivationFunction;
import evaluation.ClassificationMetrics;
import initialization.WeightInitializer;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.Adam;
import training.Trainer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Project 6 - IMDB sentiment analysis.
 *
 * Bag-of-words binary sentiment classifier. Reads
 * data/imdb/pos/*.txt and data/imdb/neg/*.txt when present,
 * otherwise generates a synthetic review corpus so the demo
 * always runs.
 */
public class SentimentApp {

    private static final int SYNTHETIC_TRAIN = 4000;
    private static final int SYNTHETIC_TEST = 1000;

    private static final String[] POSITIVE = {
        "brilliant", "masterpiece", "loved", "great", "excellent",
        "beautiful", "touching", "superb", "wonderful", "best",
        "delightful", "captivating", "heartwarming", "impressive",
        "charming", "engaging", "magnificent", "enjoyable",
        "favorite", "outstanding", "smart", "moving", "funny",
        "clever", "gorgeous"
    };

    private static final String[] NEGATIVE = {
        "boring", "terrible", "awful", "worst", "waste", "dull",
        "dreadful", "poor", "disappointing", "mess", "lame",
        "predictable", "annoying", "painful", "forgettable",
        "mediocre", "tedious", "weak", "ugly", "confusing",
        "amateurish", "shallow", "unfunny", "cheap", "tiresome"
    };

    private static final String[] FILLER = {
        "the", "a", "film", "movie", "story", "plot", "actors",
        "acting", "director", "scene", "scenes", "was", "is",
        "and", "but", "with", "of", "to", "in", "it", "this",
        "that", "very", "really", "quite", "some", "characters",
        "script", "music", "ending"
    };

    public static void main(String[] args) throws IOException {
        Corpus corpus = loadCorpus();

        Map<String, Integer> vocab = new HashMap<>();
        int nextId = 0;
        for (String review : corpus.trainTexts) {
            for (String word : tokenize(review)) {
                if (!vocab.containsKey(word)) {
                    vocab.put(word, nextId++);
                }
            }
        }

        int dim = vocab.size();
        float[][] trainX = new float[corpus.trainTexts.size()][];
        float[][] trainY =
            new float[corpus.trainTexts.size()][1];
        for (int i = 0; i < trainX.length; i++) {
            trainX[i] = vectorize(corpus.trainTexts.get(i), vocab);
            trainY[i][0] = corpus.trainLabels.get(i);
        }

        int testCount = corpus.testTexts.size();
        float[][] testX = new float[testCount][];
        int[] testLabels = new int[testCount];
        for (int i = 0; i < testCount; i++) {
            testX[i] = vectorize(corpus.testTexts.get(i), vocab);
            testLabels[i] = corpus.testLabels.get(i);
        }

        System.out.printf("Vocabulary: %d  train: %d  test: %d%n",
            vocab.size(), trainX.length, testCount);

        NeuralNetwork net = new NeuralNetwork(
            new int[] {vocab.size(), 128, 1},
            new ActivationFunction[] {
                ActivationFunction.linear(),
                ActivationFunction.relu(),
                ActivationFunction.sigmoid()
            },
            LossFunction.binaryCrossEntropy(),
            new Adam(0.001f),
            42L,
            WeightInitializer.heNormal());

        net.fit(new Trainer(10, 64, true, 42L), trainX, trainY);

        int[] predicted = new int[testCount];
        for (int i = 0; i < testCount; i++) {
            predicted[i] =
                net.predict(testX[i])[0] >= 0.5f ? 1 : 0;
        }

        int[][] confusion = ClassificationMetrics.confusionMatrix(
            predicted, testLabels, 2);

        System.out.printf("%nTest accuracy: %.2f%%%n",
            ClassificationMetrics.accuracy(predicted, testLabels)
                * 100.0f);
        ClassificationMetrics.printReport(System.out, confusion,
            new String[] {"negative", "positive"});
    }

    private static List<String> tokenize(String review) {
        List<String> words = new ArrayList<>();
        for (String word : review.toLowerCase().split("\\W+")) {
            if (!word.isBlank()) {
                words.add(word);
            }
        }
        return words;
    }

    private static float[] vectorize(String review,
                                     Map<String, Integer> vocab) {
        float[] vector = new float[vocab.size()];
        for (String word : tokenize(review)) {
            Integer id = vocab.get(word);
            if (id != null) {
                vector[id] += 1.0f;
            }
        }
        for (int i = 0; i < vector.length; i++) {
            if (vector[i] > 0.0f) {
                vector[i] = (float) Math.log1p(vector[i]);
            }
        }
        return vector;
    }

    // ── corpus loading ──────────────────────────────────────

    private static Corpus loadCorpus() throws IOException {
        Path posDir = Paths.get("data", "imdb", "pos");
        Path negDir = Paths.get("data", "imdb", "neg");
        if (Files.isDirectory(posDir) && Files.isDirectory(negDir)) {
            System.out.println("Loading reviews from "
                + posDir.getParent() + " ...");
            return fromDirectory(posDir, negDir);
        }
        System.out.println(
            "No data/imdb/pos or data/imdb/neg found - using a "
            + "synthetic review corpus.");
        return synthetic();
    }

    private static Corpus fromDirectory(Path posDir, Path negDir)
            throws IOException {
        List<String> texts = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        readInto(posDir, 1, texts, labels);
        readInto(negDir, 0, texts, labels);
        return split(texts, labels);
    }

    private static void readInto(Path dir, int label,
                                 List<String> texts,
                                 List<Integer> labels)
            throws IOException {
        try (Stream<Path> files = Files.list(dir)) {
            for (Path file : (Iterable<Path>) files::iterator) {
                if (Files.isRegularFile(file)) {
                    texts.add(Files.readString(file));
                    labels.add(label);
                }
            }
        }
    }

    private static Corpus split(List<String> texts,
                                List<Integer> labels) {
        Integer[] order = new Integer[texts.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Random rand = new Random(42L);
        for (int i = order.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            Integer t = order[i];
            order[i] = order[j];
            order[j] = t;
        }

        Corpus corpus = new Corpus();
        int cut = (int) (texts.size() * 0.8);
        for (int i = 0; i < cut; i++) {
            corpus.trainTexts.add(texts.get(order[i]));
            corpus.trainLabels.add(labels.get(order[i]));
        }
        for (int i = cut; i < texts.size(); i++) {
            corpus.testTexts.add(texts.get(order[i]));
            corpus.testLabels.add(labels.get(order[i]));
        }
        return corpus;
    }

    private static Corpus synthetic() {
        Random rand = new Random(42L);
        Corpus corpus = new Corpus();
        for (int i = 0; i < SYNTHETIC_TRAIN; i++) {
            int label = rand.nextBoolean() ? 1 : 0;
            corpus.trainTexts.add(generate(rand, label));
            corpus.trainLabels.add(label);
        }
        for (int i = 0; i < SYNTHETIC_TEST; i++) {
            int label = rand.nextBoolean() ? 1 : 0;
            corpus.testTexts.add(generate(rand, label));
            corpus.testLabels.add(label);
        }
        return corpus;
    }

    private static String generate(Random rand, int label) {
        String[] own = label == 1 ? POSITIVE : NEGATIVE;
        String[] other = label == 1 ? NEGATIVE : POSITIVE;
        int length = 20 + rand.nextInt(21);
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < length; i++) {
            double roll = rand.nextDouble();
            String word;
            if (roll < 0.35) {
                word = own[rand.nextInt(own.length)];
            } else if (roll < 0.85) {
                word = FILLER[rand.nextInt(FILLER.length)];
            } else {
                word = other[rand.nextInt(other.length)];
            }
            if (sb.length() > 0) {
                sb.append(' ');
            }
            sb.append(word);
        }
        return sb.toString();
    }

    private static final class Corpus {
        final List<String> trainTexts = new ArrayList<>();
        final List<Integer> trainLabels = new ArrayList<>();
        final List<String> testTexts = new ArrayList<>();
        final List<Integer> testLabels = new ArrayList<>();
    }
}
