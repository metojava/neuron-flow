package layers;

import activations.ActivationFunction;
import initialization.WeightInitializer;
import losses.LossFunction;
import optim.SGD;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression test: networks using activations whose derivative
 * needs the pre-activation (GELU, Swish/SiLU) must train without
 * throwing and must reduce the loss.
 */
class SmoothActivationTrainingTest {

    private static final float[][] INPUTS = {
        {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
    };
    private static final float[][] TARGETS = {
        {0.1f}, {0.9f}, {0.9f}, {0.1f}
    };

    private static void assertTrainsWith(ActivationFunction hidden) {
        ActivationFunction[] acts = {
            ActivationFunction.linear(), hidden,
            ActivationFunction.sigmoid()
        };
        NeuralNetwork net = new NeuralNetwork(
            new int[] {2, 8, 1}, acts,
            LossFunction.meanSquaredError(),
            new SGD(0.5f), 7L,
            WeightInitializer.xavierUniform());

        float before = net.calculateLoss(INPUTS, TARGETS);
        for (int epoch = 0; epoch < 500; epoch++) {
            net.trainBatch(INPUTS, TARGETS);
        }
        float after = net.calculateLoss(INPUTS, TARGETS);

        assertTrue(Float.isFinite(after), "loss must be finite");
        assertTrue(after < before,
            hidden.getClass().getSimpleName()
            + " did not reduce loss: " + before + " -> " + after);
    }

    @Test
    void geluNetworkTrains() {
        assertTrainsWith(ActivationFunction.gelu());
    }

    @Test
    void swishNetworkTrains() {
        assertTrainsWith(ActivationFunction.swish());
    }

    @Test
    void siluNetworkTrains() {
        assertTrainsWith(ActivationFunction.silu());
    }
}
