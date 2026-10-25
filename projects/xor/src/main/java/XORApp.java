import activations.ActivationFunction;
import initialization.WeightInitializer;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.SGD;
import training.Trainer;

/**
 * Project 1 - XOR logic gate.
 *
 * The smallest possible nonlinear problem: four examples that no
 * single neuron can separate. A 2-4-1 MLP with sigmoid hidden and
 * output units learns the truth table from data alone.
 *
 * Run with: mvn -q exec:exec
 */
public class XORApp {

    public static void main(String[] args) {
        float[][] inputs = {
            {0.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 0.0f},
            {1.0f, 1.0f}
        };
        float[][] targets = {
            {0.0f}, {1.0f}, {1.0f}, {0.0f}
        };

        NeuralNetwork net = new NeuralNetwork(
            new int[] {2, 4, 1},
            new ActivationFunction[] {
                ActivationFunction.linear(),
                ActivationFunction.sigmoid(),
                ActivationFunction.sigmoid()
            },
            LossFunction.meanSquaredError(),
            new SGD(0.5f),
            42L,
            WeightInitializer.xavierUniform());

        System.out.println("epoch    loss");
        for (int epoch = 1; epoch <= 2000; epoch++) {
            net.fit(new Trainer(1, 4, true, epoch), inputs, targets);

            if (epoch % 400 == 0) {
                System.out.printf("%5d    %.4f%n", epoch,
                    net.calculateLoss(inputs, targets));
            }
        }

        System.out.println();
        for (float[] row : inputs) {
            float y = net.predict(row)[0];
            System.out.printf("%d XOR %d = %.3f%n",
                (int) row[0], (int) row[1], y);
        }
    }
}
