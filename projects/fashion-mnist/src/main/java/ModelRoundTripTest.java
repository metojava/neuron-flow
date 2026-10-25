import activations.ActivationFunction;
import data.FashionMNISTLoader;
import data.FashionMNISTDownloader;
import data.FashionMNISTLoader.Dataset;
import evaluation.ClassificationMetrics;
import initialization.WeightInitializer;
import layers.ModelFormat;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.Adam;

import java.nio.file.Path;

/**
 * Verifies that a model saved in BINARY, JSON and CSV format
 * produces identical predictions to the in-memory model.
 */
public class ModelRoundTripTest {

    private static final ActivationFunction[] ACTIVATIONS = {
        ActivationFunction.linear(),
        ActivationFunction.relu(),
        ActivationFunction.linear()
    };

    public static void main(String[] args) throws Exception {
        Path dir = FashionMNISTDownloader.ensureDownloaded();
        Dataset data = FashionMNISTLoader.load(dir);

        NeuralNetwork bin = NeuralNetwork.loadModel(
            Path.of("model/fashion.bin"), ModelFormat.BINARY,
            ACTIVATIONS, LossFunction.softmaxCrossEntropy(),
            new Adam(0.001f), 123L, WeightInitializer.heNormal());
        NeuralNetwork json = NeuralNetwork.loadModel(
            Path.of("model/fashion.json"), ModelFormat.JSON,
            ACTIVATIONS, LossFunction.softmaxCrossEntropy(),
            new Adam(0.001f), 123L, WeightInitializer.heNormal());
        NeuralNetwork csv = NeuralNetwork.loadModel(
            Path.of("model/fashion.csv"), ModelFormat.CSV,
            ACTIVATIONS, LossFunction.softmaxCrossEntropy(),
            new Adam(0.001f), 123L, WeightInitializer.heNormal());

        int limit = 500;
        int matchJson = 0, matchCsv = 0;

        for (int i = 0; i < limit; i++) {
            int expected = ClassificationMetrics.argmax(
                bin.predict(data.testImages()[i]));
            if (ClassificationMetrics.argmax(
                    json.predict(data.testImages()[i]))
                    == expected) matchJson++;
            if (ClassificationMetrics.argmax(
                    csv.predict(data.testImages()[i]))
                    == expected) matchCsv++;
        }

        System.out.printf(
            "Binary vs JSON: %d/%d match (%.1f%%)%n",
            matchJson, limit, 100.0 * matchJson / limit);
        System.out.printf(
            "Binary vs CSV:  %d/%d match (%.1f%%)%n",
            matchCsv, limit, 100.0 * matchCsv / limit);

        if (matchJson == limit && matchCsv == limit) {
            System.out.println("ROUND-TRIP OK");
        } else {
            System.out.println("ROUND-TRIP MISMATCH");
            System.exit(1);
        }
    }
}
