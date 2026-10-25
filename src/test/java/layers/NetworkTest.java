package layers;

import activations.ActivationFunction;
import losses.LossFunction;
import optim.SGD;
import org.junit.jupiter.api.Test;
import initialization.WeightInitializer;

import static org.junit.jupiter.api.Assertions.*;

class NetworkTest {
    @Test
    void xor() {
        NeuralNetwork net = new NeuralNetwork(
            new int[]{2, 4, 1},
            new ActivationFunction[]{
                ActivationFunction.linear(),
                ActivationFunction.tanh(),
                ActivationFunction.tanh()
            },
            LossFunction.meanSquaredError(),
            new SGD(0.1f),
            123L,
            WeightInitializer.xavierUniform()
        );

        float[][] X = {
            {0,0},{0,1},{1,0},{1,1}
        };
        float[][] y = {
            {0},{1},{1},{0}
        };

        training.Trainer trainer =
            new training.Trainer(5000, 4, true, 123L);
        net.fit(trainer, X, y);

        for (int i = 0; i < X.length; i++) {
            float[] pred = net.forwardPropagate(X[i]);
            assertTrue(pred[0] > 0.5f && i == 1 || pred[0] > 0.5f && i == 2
                    || pred[0] < 0.5f && i == 0 || pred[0] < 0.5f && i == 3);
        }
    }
}
