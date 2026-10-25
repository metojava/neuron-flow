import data.FashionMNISTDownloader;
import data.FashionMNISTLoader;
import evaluation.ClassificationMetrics;
import layers.DenseClassifier;
import optim.Adam;
import training.DataParallelTrainer;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

/**
 * Keras starter for Fashion-MNIST, using the library's dense classifier:
 *
 * <pre>
 *   model = tf.keras.Sequential([
 *       tf.keras.layers.Flatten(input_shape=(28, 28)),
 *       tf.keras.layers.Dense(128, activation='relu'),
 *       tf.keras.layers.Dense(10)
 *   ])
 *   model.compile(optimizer='adam',
 *                 loss=SparseCategoricalCrossentropy(from_logits=True))
 *   model.fit(x_train, y_train, epochs=10, batch_size=32)
 * </pre>
 *
 * The loader already flattens images to 784 values, so Flatten is the
 * 784-unit input. Training uses Adam (learning rate 0.001) on the flat
 * parameter vector, with mini-batches split across worker threads.
 * Weights use He-normal initialization (Keras uses Glorot-uniform).
 *
 * Arguments, in the same order as the pom's exec:exec properties:
 *   [trainLimit] [epochs] [batchSize] [testLimit] [compact (ignored)] [threads]
 * Defaults: 60000 10 32 10000, threads = min(4, available processors).
 *
 * With Maven: mvn compile exec:exec -DmainClass=FashionMnistKerasLike
 *             -DrunArgs=60000 -DcnnEpochs=10 -DbatchSize=32 -DtestLimit=10000
 */
public final class FashionMnistKerasLike {

    private static final long SEED = 123L;
    private static final float LEARNING_RATE = 0.001f;

    private FashionMnistKerasLike() {
    }

    public static void main(String[] args) throws Exception {
        int trainLimit = intArg(args, 0, 60000);
        int epochs = intArg(args, 1, 10);
        int batchSize = intArg(args, 2, 32);
        int testLimit = intArg(args, 3, 10000);
        int threads = intArg(args, 5,
            Math.min(4, Runtime.getRuntime().availableProcessors()));
        if (trainLimit <= 0 || epochs <= 0 || batchSize <= 0
            || testLimit <= 0 || threads <= 0) {
            throw new IllegalArgumentException(
                "All numeric arguments must be positive");
        }

        Path dir = FashionMNISTDownloader.ensureDownloaded();
        FashionMNISTLoader.Dataset data = FashionMNISTLoader.load(dir);

        int trainCount = Math.min(trainLimit, data.trainImages().length);
        int testCount = Math.min(testLimit, data.testLabels().length);
        float[][] xTrain = Arrays.copyOf(data.trainImages(), trainCount);
        int[] yTrain = Arrays.copyOf(data.trainLabels(), trainCount);
        float[][] xTest = Arrays.copyOf(data.testImages(), testCount);
        int[] yTest = Arrays.copyOf(data.testLabels(), testCount);

        // Flatten(input_shape=(28, 28)) -> Dense(128, relu) -> Dense(10)
        int[] sizes = {FashionMNISTLoader.INPUT_SIZE, 128,
                       FashionMNISTLoader.NUM_CLASSES};
        DenseClassifier model = new DenseClassifier(SEED, sizes,
            new Adam(LEARNING_RATE));

        System.out.println("Model: Flatten(784) -> Dense(128, relu) -> Dense(10)");
        System.out.println("Parameters: " + model.parameterCount());
        System.out.printf("Training on %d images, testing on %d images, "
            + "%d worker threads%n", trainCount, testCount, threads);
        System.out.println();
        System.out.printf("%-7s %-12s %-12s %s%n",
            "Epoch", "Loss", "Test acc", "Time");
        System.out.println("----------------------------------------");

        int[] order = new int[trainCount];
        for (int i = 0; i < trainCount; i++) {
            order[i] = i;
        }
        Random random = new Random(SEED);

        try (DataParallelTrainer trainer = new DataParallelTrainer(model, threads)) {
            for (int epoch = 1; epoch <= epochs; epoch++) {
                shuffle(order, random);            // shuffle=True

                long start = System.nanoTime();
                double loss = trainer.trainEpoch(xTrain, yTrain, order,
                    trainCount, batchSize, LEARNING_RATE);
                double seconds = (System.nanoTime() - start) / 1e9;

                int[] predicted = predict(model, xTest, testCount);
                float acc = ClassificationMetrics.accuracy(predicted, yTest);

                System.out.printf("%-7d %-12.6f %-12s %.1f s%n",
                    epoch, loss, String.format("%.2f%%", acc * 100), seconds);
            }
        }

        int[] predicted = predict(model, xTest, testCount);
        int correct = 0;
        for (int i = 0; i < testCount; i++) {
            if (predicted[i] == yTest[i]) {
                correct++;
            }
        }
        System.out.println();
        System.out.printf("Test accuracy: %.2f%% (%d/%d)%n",
            100.0 * correct / testCount, correct, testCount);
        int[][] confusion = ClassificationMetrics.confusionMatrix(
            predicted, yTest, FashionMNISTLoader.NUM_CLASSES);
        ClassificationMetrics.printReport(System.out, confusion,
            FashionMNISTLoader.CLASS_NAMES);
    }

    private static int[] predict(DenseClassifier model, float[][] images,
                                 int count) {
        int[] result = new int[count];
        for (int i = 0; i < count; i++) {
            result[i] = model.predict(images[i]);
        }
        return result;
    }

    private static void shuffle(int[] order, Random random) {
        for (int i = order.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
    }

    private static int intArg(String[] args, int index, int fallback) {
        return args.length > index ? Integer.parseInt(args[index]) : fallback;
    }
}
