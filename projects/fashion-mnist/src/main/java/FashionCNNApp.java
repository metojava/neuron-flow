import data.FashionMNISTDownloader;
import data.FashionMNISTLoader;
import evaluation.ClassificationMetrics;
import layers.ConvClassifier;
import training.DataParallelTrainer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

/**
 * Trainable Fashion-MNIST classifier:
 * Conv2D(1,8,3) -> ReLU -> MaxPool2D -> Conv2D(8,16,3)
 * -> ReLU -> MaxPool2D -> Flatten -> Dense(128) -> Dense(10).
 *
 * Optional args: trainLimit, epochs, batchSize, testLimit, compact, threads.
 * Run with --add-modules jdk.incubator.vector.
 */
public final class FashionCNNApp {
    private FashionCNNApp() {
    }

    public static void main(String[] args) throws IOException {
        int trainLimit = args.length > 0 ? Integer.parseInt(args[0]) : 60000;
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : 1;
        int batchSize = args.length > 2 ? Integer.parseInt(args[2]) : 128;
        int testLimit = args.length > 3 ? Integer.parseInt(args[3]) : 10000;
        boolean compact = args.length > 4 ? Boolean.parseBoolean(args[4]) : true;
        int threads = args.length > 5 ? Integer.parseInt(args[5])
            : Math.min(4, Runtime.getRuntime().availableProcessors());
        if (trainLimit <= 0 || epochs <= 0 || batchSize <= 0
            || testLimit <= 0) {
            throw new IllegalArgumentException("Training settings must be positive");
        }

        long runStart = System.nanoTime();
        Path directory = FashionMNISTDownloader.ensureDownloaded();
        FashionMNISTLoader.Dataset data = FashionMNISTLoader.load(directory);
        int count = Math.min(trainLimit, data.trainImages().length);
        int side = (int) Math.round(Math.sqrt(data.trainImages()[0].length));
        ConvClassifier net = compact
            ? ConvClassifier.compact(123L, side, FashionMNISTLoader.NUM_CLASSES)
            : new ConvClassifier(123L, side, FashionMNISTLoader.NUM_CLASSES);
        int[] order = new int[count];
        for (int i = 0; i < count; i++) order[i] = i;
        Random random = new Random(123L);
        float learningRate = 0.01f;

        System.out.printf("Conv2D model (%s): %d parameters, %d training images, batch size %d%n",
            compact ? "compact" : "full", net.parameterCount(), count, batchSize);
        long trainingStart = System.nanoTime();
        try (DataParallelTrainer trainer = new DataParallelTrainer(net, threads)) {
            System.out.printf("Worker threads: %d%n", threads);
            for (int epoch = 1; epoch <= epochs; epoch++) {
                long epochStart = System.nanoTime();
                shuffle(order, random);
                double loss = trainer.trainEpoch(data.trainImages(), data.trainLabels(),
                    order, count, batchSize, learningRate);
                System.out.printf("Epoch %d/%d, mean loss %.4f, time %.2f s%n",
                    epoch, epochs, loss, secondsSince(epochStart));
            }
        }
        System.out.printf("Training time: %.2f s%n",
            secondsSince(trainingStart));

        long evaluationStart = System.nanoTime();
        int evalCount = Math.min(testLimit, data.testLabels().length);
        int[] actual = java.util.Arrays.copyOf(data.testLabels(), evalCount);
        int[] predicted = new int[actual.length];
        for (int i = 0; i < predicted.length; i++) {
            predicted[i] = net.predict(data.testImages()[i]);
        }
        System.out.printf("Test accuracy: %.2f%%%n",
            100.0f * ClassificationMetrics.accuracy(predicted, actual));
        int[][] confusion = ClassificationMetrics.confusionMatrix(
            predicted, actual, FashionMNISTLoader.NUM_CLASSES);
        ClassificationMetrics.printReport(System.out, confusion,
            FashionMNISTLoader.CLASS_NAMES);
        System.out.printf("Evaluation time: %.2f s%n",
            secondsSince(evaluationStart));
        System.out.printf("Total time (including data load): %.2f s%n",
            secondsSince(runStart));
    }

    private static double secondsSince(long start) {
        return (System.nanoTime() - start) / 1_000_000_000.0;
    }

    private static void shuffle(int[] order, Random random) {
        for (int i = order.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
    }
}
