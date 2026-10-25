package org.example.deeplearning.examples;

import org.example.deeplearning.activation.ActivationFunction;
import org.example.deeplearning.loss.LossFunction;
import org.example.deeplearning.optim.Adam;
import org.example.deeplearning.training.Trainer;
import org.example.deeplearning.core.NeuralNetwork;

public class Main {
    public static void main(String[] args) {
        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[][] targets = {
            {0},
            {1},
            {1},
            {0}
        };

        int[] topology = {2, 8, 1};

        ActivationFunction[] activations = {
            ActivationFunction.linear(),
            ActivationFunction.tanh(),
            ActivationFunction.sigmoid()
        };

        NeuralNetwork network = new NeuralNetwork(
            topology,
            activations,
            LossFunction.meanSquaredError(),
            new Adam(0.01),
            42L
        );

        Trainer trainer = new Trainer(
            10_000, // epochs
            4,      // batch size
            true,   // shuffle
            42L     // reproducible shuffle seed
        );

        System.out.printf("Initial loss: %.6f%n",
            network.calculateLoss(inputs, targets));

        network.fit(trainer, inputs, targets);

        System.out.printf("Final loss: %.6f%n",
            network.calculateLoss(inputs, targets));

        System.out.println("\nPredictions:");

        for (int i = 0; i < inputs.length; i++) {
            double prediction = network.predict(inputs[i])[0];

            System.out.printf(
                "[%.0f, %.0f] -> %.5f (target: %.0f)%n",
                inputs[i][0],
                inputs[i][1],
                prediction,
                targets[i][0]
            );
        }
    }
}