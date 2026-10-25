package layers;

import activations.ActivationFunction;
import losses.LossFunction;
import optim.SGD;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class NeuralNetworkTest {
    @TempDir Path tempDir;

    private NeuralNetwork linearNetwork() {
        return new NeuralNetwork(new int[] {2, 1},
            new ActivationFunction[] {ActivationFunction.linear(), ActivationFunction.linear()},
            LossFunction.meanSquaredError(), new SGD(0.1f), 42L);
    }

    @Test
    void batchTrainingReducesLoss() {
        NeuralNetwork network = linearNetwork();
        network.setParameters(new float[] {0, 0, 0});
        float[][] inputs = {{1, 0}, {0, 1}};
        float[][] targets = {{1}, {1}};
        float before = network.calculateLoss(inputs, targets);
        for (int i = 0; i < 50; i++) {
            network.trainBatch(inputs, targets);
        }
        assertTrue(network.calculateLoss(inputs, targets) < before * 0.01f);
    }

    @Test
    void savedModelsRoundTripInEveryFormat() throws IOException {
        NeuralNetwork network = linearNetwork();
        network.setParameters(new float[] {0.25f, -0.5f, 1.5f});
        for (String extension : new String[] {"bin", "json", "csv"}) {
            Path file = tempDir.resolve("model." + extension);
            network.saveModel(file);
            NeuralNetwork restored = NeuralNetwork.loadModel(file,
                new ActivationFunction[] {ActivationFunction.linear(), ActivationFunction.linear()},
                LossFunction.meanSquaredError(), new SGD(0.1f), 42L,
                initialization.WeightInitializer.xavierUniform());
            assertArrayEquals(network.predict(new float[] {2, 3}),
                restored.predict(new float[] {2, 3}), 1e-6f);
        }
    }
}
