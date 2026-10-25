package examples;

import activations.ActivationFunction;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.Adam;
import training.Trainer;

/**
 * Minimal multi-layer perceptron: learns XOR with one hidden ReLU layer.
 *
 * <p>Run from the project root:
 * {@code java --add-modules jdk.incubator.vector -cp target/classes examples.SimpleMlpDemo}
 */
public final class SimpleMlpDemo {

    private SimpleMlpDemo() {
    }

    public static void main(String[] args) {
        float[][] inputs = {
            {0f, 0f},
            {0f, 1f},
            {1f, 0f},
            {1f, 1f}
        };
        float[][] targets = {
            {0f},
            {1f},
            {1f},
            {0f}
        };

        NeuralNetwork mlp = new NeuralNetwork(
                new int[]{2, 8, 1},
                new ActivationFunction[]{
                        ActivationFunction.linear(),
                        ActivationFunction.relu(),
                        ActivationFunction.sigmoid()
                },
                LossFunction.meanSquaredError(),
                new Adam(0.01f),
                42L);

        mlp.fit(new Trainer(2_000, 4, true, 42L), inputs, targets);

        System.out.printf("Final loss: %.6f%n", mlp.calculateLoss(inputs, targets));
        for (int i = 0; i < inputs.length; i++) {
            float output = mlp.forwardPropagate(inputs[i])[0];
            System.out.printf("%.0f XOR %.0f = %.4f (expected %.0f)%n",
                    inputs[i][0], inputs[i][1], output, targets[i][0]);
        }
    }
}
