package layers;

import activations.ActivationFunction;
import losses.LossFunction;
import java.util.Arrays;

/**
 * A simple sequential model builder that wraps NeuralNetwork-like
 * configuration with a more familiar API.
 */
public class Sequential {

    private final java.util.ArrayList<LayerConfig> configs =
        new java.util.ArrayList<LayerConfig>();

    public Sequential() {
    }

    public Sequential add(int units) {
        configs.add(new LayerConfig(units, null));
        return this;
    }

    public Sequential add(int units, ActivationFunction activation) {
        configs.add(new LayerConfig(units, activation));
        return this;
    }

    public NeuralNetwork build(float learningRate) {
        return build(activations(), LossFunction.meanSquaredError(),
                     new optim.SGD(learningRate), 42L,
                     initialization.WeightInitializer.xavierUniform());
    }

    public NeuralNetwork build(ActivationFunction[] activations,
                               losses.LossFunction lossFunction,
                               optim.Optimizer optimizer,
                               long seed,
                               initialization.WeightInitializer initializer) {
        if (configs.isEmpty()) {
            throw new IllegalArgumentException("No layers added");
        }

        int[] topology = new int[configs.size()];
        for (int i = 0; i < configs.size(); i++) {
            topology[i] = configs.get(i).units;
        }

        if (activations.length != topology.length) {
            throw new IllegalArgumentException(
                "Activations length must match topology length");
        }

        return new NeuralNetwork(topology, activations, lossFunction,
                                 optimizer, seed, initializer);
    }

    public ActivationFunction[] activations() {
        ActivationFunction[] acts =
            new ActivationFunction[configs.size()];
        for (int i = 0; i < configs.size(); i++) {
            LayerConfig cfg = configs.get(i);
            if (i == 0) {
                acts[i] = ActivationFunction.linear();
            } else if (cfg.activation != null) {
                acts[i] = cfg.activation;
            } else {
                acts[i] = ActivationFunction.relu();
            }
        }
        return acts;
    }

    private static class LayerConfig {
        int units;
        ActivationFunction activation;

        LayerConfig(int units, ActivationFunction activation) {
            this.units = units;
            this.activation = activation;
        }
    }
}
