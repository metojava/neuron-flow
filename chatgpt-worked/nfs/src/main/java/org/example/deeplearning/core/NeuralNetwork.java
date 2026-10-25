package org.example.deeplearning.core;

import org.example.deeplearning.activation.ActivationFunction;
import org.example.deeplearning.loss.LossFunction;
import org.example.deeplearning.initialization.WeightInitializer;
import org.example.deeplearning.optim.Optimizer;
import org.example.deeplearning.optim.SGD;
import org.example.deeplearning.training.Trainer;

import java.util.Arrays;
import java.util.Random;

public class NeuralNetwork {
    private final Layer[] layers;
    private final ActivationFunction[] activations;
    private final LossFunction lossFunction;
    private final Optimizer optimizer;

    public NeuralNetwork(int[] topology, double learningRate) {
        this(topology, repeatedSigmoid(topology),
             LossFunction.meanSquaredError(),
             new SGD(learningRate), 42L);
    }

    public NeuralNetwork(
            int[] topology,
            ActivationFunction[] activations,
            LossFunction lossFunction,
            Optimizer optimizer,
            long seed) {
        this(topology, activations, lossFunction, optimizer, seed, WeightInitializer.xavierUniform());
    }

    public NeuralNetwork(int[] topology, ActivationFunction[] activations,
            LossFunction lossFunction, Optimizer optimizer, long seed,
            WeightInitializer initializer) {

        if (initializer == null) throw new IllegalArgumentException("Initializer must not be null");
        if (topology == null || topology.length < 2) {
            throw new IllegalArgumentException(
                "Topology must contain input and output layers");
        }

        for (int size : topology) {
            if (size <= 0) {
                throw new IllegalArgumentException(
                    "Layer sizes must be positive");
            }
        }

        if (activations == null ||
            activations.length != topology.length ||
            lossFunction == null || optimizer == null) {
            throw new IllegalArgumentException(
                "Invalid network configuration");
        }

        this.activations = activations.clone();
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.layers = new Layer[topology.length];

        Random rand = new Random(seed);
        layers[0] = new Layer(topology[0], 0, rand);

        // Xavier-style initialization for sigmoid/tanh-friendly networks.
        for (int l = 1; l < topology.length; l++) {
            int fanIn = topology[l - 1];
            int fanOut = topology[l];
            layers[l] = new Layer(topology[l], fanIn, rand);

            for (Neuron neuron : layers[l].neurons) {
                for (int j = 0; j < neuron.weights.length; j++) {
                    neuron.weights[j] = initializer.sample(fanIn, fanOut, rand);
                }

                neuron.bias = 0.0;
            }
        }
    }

    private static ActivationFunction[] repeatedSigmoid(
            int[] topology) {
        if (topology == null || topology.length < 2) {
            throw new IllegalArgumentException("Invalid topology");
        }

        ActivationFunction[] result =
            new ActivationFunction[topology.length];

        result[0] = ActivationFunction.linear();

        for (int i = 1; i < result.length; i++) {
            result[i] = ActivationFunction.sigmoid();
        }

        return result;
    }

    public double[] forwardPropagate(double[] inputs) {
        if (inputs == null ||
            inputs.length != layers[0].neurons.length) {
            throw new IllegalArgumentException(
                "Input dimension does not match network");
        }

        for (int i = 0; i < inputs.length; i++) {
            if (!Double.isFinite(inputs[i])) {
                throw new IllegalArgumentException(
                    "Inputs must be finite");
            }

            layers[0].neurons[i].value = inputs[i];
        }

        for (int l = 1; l < layers.length; l++) {
            Layer previous = layers[l - 1];
            Layer current = layers[l];

            for (Neuron neuron : current.neurons) {
                double sum = neuron.bias;

                for (int j = 0; j < previous.neurons.length; j++) {
                    sum += previous.neurons[j].value
                         * neuron.weights[j];
                }

                neuron.value = activations[l].activate(sum);
            }
        }

        Layer output = layers[layers.length - 1];
        double[] result = new double[output.neurons.length];

        for (int i = 0; i < result.length; i++) {
            result[i] = output.neurons[i].value;
        }

        return result;
    }

    public double[] predict(double[] inputs) {
        return forwardPropagate(inputs);
    }

    public double calculateLoss(double[][] inputs,
                                double[][] targets) {
        validateDataset(inputs, targets);
        double total = 0.0;

        for (int i = 0; i < inputs.length; i++) {
            total += lossFunction.loss(
                forwardPropagate(inputs[i]), targets[i]);
        }

        return total / inputs.length;
    }

    // Trains on one mini-batch. Gradients are accumulated before
    // the optimizer updates any parameters.
    public void trainBatch(double[][] inputs, double[][] targets) {
        validateDataset(inputs, targets);

        int parameterCount = parameterCount();
        double[] parameters = flattenParameters();
        double[] gradients = new double[parameterCount];

        for (int sample = 0; sample < inputs.length; sample++) {
            double[] prediction = forwardPropagate(inputs[sample]);

            if (targets[sample].length != prediction.length) {
                throw new IllegalArgumentException(
                    "Target dimension does not match output");
            }

            double[] outputGradient =
                lossFunction.gradient(prediction, targets[sample]);

            Layer outputLayer = layers[layers.length - 1];

            for (int i = 0; i < outputLayer.neurons.length; i++) {
                Neuron neuron = outputLayer.neurons[i];
                neuron.gradient = outputGradient[i]
                    * activations[layers.length - 1]
                        .derivativeFromOutput(neuron.value);
            }

            // Calculate hidden-layer gradients backwards.
            for (int l = layers.length - 2; l > 0; l--) {
                Layer current = layers[l];
                Layer next = layers[l + 1];

                for (int i = 0; i < current.neurons.length; i++) {
                    double sum = 0.0;

                    for (Neuron nextNeuron : next.neurons) {
                        sum += nextNeuron.gradient
                             * nextNeuron.weights[i];
                    }

                    Neuron neuron = current.neurons[i];
                    neuron.gradient = sum
                        * activations[l].derivativeFromOutput(
                            neuron.value);
                }
            }

            // Accumulate parameter gradients in exactly the same
            // order used by flattenParameters().
            int k = 0;

            for (int l = 1; l < layers.length; l++) {
                Layer previous = layers[l - 1];

                for (Neuron neuron : layers[l].neurons) {
                    for (int j = 0; j < neuron.weights.length; j++) {
                        gradients[k++] += neuron.gradient
                            * previous.neurons[j].value;
                    }

                    gradients[k++] += neuron.gradient; // bias
                }
            }
        }

        for (int i = 0; i < gradients.length; i++) {
            gradients[i] /= inputs.length;
        }

        optimizer.update(parameters, gradients);
        unflattenParameters(parameters);
    }

    public void fit(Trainer trainer,
                    double[][] inputs,
                    double[][] targets) {
        validateDataset(inputs, targets);

        int[] order = new int[inputs.length];

        for (int epoch = 0; epoch < trainer.getEpochs(); epoch++) {
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }

            if (trainer.shouldShuffle()) {
                Random random = trainer.getRandom();

                for (int i = order.length - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    int temp = order[i];
                    order[i] = order[j];
                    order[j] = temp;
                }
            }

            int batchSize = trainer.getBatchSize();

            for (int start = 0; start < order.length;
                 start += batchSize) {
                int end = Math.min(start + batchSize, order.length);
                int count = end - start;

                double[][] batchInputs = new double[count][];
                double[][] batchTargets = new double[count][];

                for (int i = 0; i < count; i++) {
                    batchInputs[i] = inputs[order[start + i]];
                    batchTargets[i] = targets[order[start + i]];
                }

                trainBatch(batchInputs, batchTargets);
            }
        }
    }

    private int parameterCount() {
        int count = 0;

        for (int l = 1; l < layers.length; l++) {
            for (Neuron neuron : layers[l].neurons) {
                count += neuron.weights.length + 1;
            }
        }

        return count;
    }

    private double[] flattenParameters() {
        double[] result = new double[parameterCount()];
        int k = 0;

        for (int l = 1; l < layers.length; l++) {
            for (Neuron neuron : layers[l].neurons) {
                for (double weight : neuron.weights) {
                    result[k++] = weight;
                }

                result[k++] = neuron.bias;
            }
        }

        return result;
    }

    private void unflattenParameters(double[] parameters) {
        int k = 0;

        for (int l = 1; l < layers.length; l++) {
            for (Neuron neuron : layers[l].neurons) {
                for (int j = 0; j < neuron.weights.length; j++) {
                    neuron.weights[j] = parameters[k++];
                }

                neuron.bias = parameters[k++];
            }
        }
    }

    private void validateDataset(double[][] inputs,
                                 double[][] targets) {
        if (inputs == null || targets == null ||
            inputs.length == 0 || inputs.length != targets.length) {
            throw new IllegalArgumentException(
                "Dataset must have matching, nonempty inputs and targets");
        }

        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] == null ||
                inputs[i].length != layers[0].neurons.length ||
                targets[i] == null ||
                targets[i].length != layers[layers.length - 1]
                                               .neurons.length) {
                throw new IllegalArgumentException(
                    "Invalid input or target dimensions at row " + i);
            }
        }
    }
}