package org.example.deeplearning.examples;
import org.example.deeplearning.activation.ActivationFunction;
import org.example.deeplearning.core.NeuralNetwork;
import org.example.deeplearning.initialization.WeightInitializer;
import org.example.deeplearning.loss.HuberLoss;
import org.example.deeplearning.optim.RMSProp;
import org.example.deeplearning.training.Trainer;
/** Fit y = 2x + 1. */
public final class RegressionExample {
    public static void main(String[] args) {
        double[][] x = {{-2}, {-1}, {0}, {1}, {2}};
        double[][] y = {{-3}, {-1}, {1}, {3}, {5}};
        NeuralNetwork net = new NeuralNetwork(new int[]{1, 1},
            new ActivationFunction[]{ActivationFunction.linear(), ActivationFunction.linear()},
            new HuberLoss(1), new RMSProp(0.01), 42L, WeightInitializer.xavierUniform());
        System.out.printf("Before: %.6f%n", net.calculateLoss(x, y));
        net.fit(new Trainer(1000, 5, true, 42L), x, y);
        System.out.printf("After: %.6f%n", net.calculateLoss(x, y));
        System.out.printf("x=1.5: %.4f (target 4.0)%n", net.predict(new double[]{1.5})[0]);
    }
}
