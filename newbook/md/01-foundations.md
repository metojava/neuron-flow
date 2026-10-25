# Neuron, Layer, and the neural-network constructor

*Every network in this project is built from three plain Java classes. We read them one by one: what each field means, what each method does, and how they fit together.*

## 1.1 The `Neuron` class

A neuron stores the numbers a network learns: one **weight per input** plus a **bias**. It also caches two values used during training: the current activation (`value`) and the error signal flowing back into it (`gradient`).

```
public class Neuron {
    public double[] weights;   // one weight per input to this neuron
    public double bias;        // additive offset
    public double value;       // cached activation (a) after forward pass
    public double gradient;    // cached δ used during backpropagation

    /**
     * Creates a neuron that expects {@code numInputs} inputs.
     *
     * @param numInputs how many weights to allocate (one per upstream neuron)
     * @param rand      source of randomness so initialization is reproducible
     *                  when the caller passes a seeded Random
     */
    public Neuron(int numInputs, Random rand) {
        this.weights = new double[numInputs];

        for (int i = 0; i < numInputs; i++) {
            // Uniform in [-1, 1); overwritten later with Xavier scaling.
            this.weights[i] = rand.nextDouble() * 2.0 - 1.0;
        }

        // Uniform in [-1, 1); overwritten to 0.0 by the network constructor.
        this.bias = rand.nextDouble() * 2.0 - 1.0;
    }
}
```

### Field reference

| Field | Type | Meaning |
| --- | --- | --- |
| `weights` | `double[]` | One weight per input. Length equals the size of the previous layer. |
| `bias` | `double` | Constant added to the weighted sum before the activation. |
| `value` | `double` | The activation `a = f(z)` computed in the last forward pass. |
| `gradient` | `double` | The backprop error signal `δ` for this neuron. |

> **Why public fields?** This library favors clarity over encapsulation. The optimizer needs to read and write every weight, and the forward/backprop code touches `value` and `gradient` constantly. Public fields keep the math readable.

## 1.2 The `Layer` class

A layer is just a fixed-size array of neurons that all share the same number of inputs.

```
public class Layer {
    public Neuron[] neurons;

    /**
     * Creates a layer of {@code numNeurons} neurons, each expecting
     * {@code numInputsPerNeuron} inputs.
     *
     * @param numNeurons         how many neurons in this layer
     * @param numInputsPerNeuron how many weights each neuron receives
     * @param rand               shared Random so the whole network is
     *                           reproducible from one seed
     */
    public Layer(int numNeurons, int numInputsPerNeuron, Random rand) {
        this.neurons = new Neuron[numNeurons];

        for (int i = 0; i < numNeurons; i++) {
            this.neurons[i] = new Neuron(numInputsPerNeuron, rand);
        }
    }
}
```

The input layer is special: it has no weights, so its neurons are created with `numInputsPerNeuron = 0`. Their `value` field is simply set to the raw input.

## 1.3 The `NeuralNetwork` constructor

The network holds an array of layers, one activation per layer, a loss function, and an optimizer. The constructor validates the configuration, builds the layers, and applies **Xavier initialization** to every trainable weight.

```
public class NeuralNetwork {
    private final Layer[] layers;
    private final ActivationFunction[] activations;
    private final LossFunction lossFunction;
    private final Optimizer optimizer;

    /**
     * Full constructor.
     *
     * @param topology      layer sizes, e.g. {784, 128, 10}; index 0 is input
     * @param activations   one activation per layer; input layer uses linear
     * @param lossFunction  how to measure error (e.g. mean squared error)
     * @param optimizer     how to update weights from gradients (e.g. Adam)
     * @param seed          RNG seed so runs are reproducible
     */
    public NeuralNetwork(
            int[] topology,
            ActivationFunction[] activations,
            LossFunction lossFunction,
            Optimizer optimizer,
            long seed) {

        // --- validation ---
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

        // Input layer: neurons hold values only, no weights.
        layers[0] = new Layer(topology[0], 0, rand);

        // Hidden + output layers: Xavier-style initialization.
        for (int l = 1; l < topology.length; l++) {
            int fanIn  = topology[l - 1];
            int fanOut = topology[l];
            layers[l] = new Layer(topology[l], fanIn, rand);

            double limit = Math.sqrt(6.0 / (fanIn + fanOut));

            for (Neuron neuron : layers[l].neurons) {
                for (int j = 0; j < neuron.weights.length; j++) {
                    neuron.weights[j] =
                        (rand.nextDouble() * 2.0 - 1.0) * limit;
                }
                neuron.bias = 0.0;   // biases start at zero
            }
        }
    }
}
```

### Xavier initialization, briefly

Starting weights that are too large make activations explode; too small and they vanish. Xavier's rule picks a range `±√(6/(fanIn+fanOut))` that keeps variance stable across layers. Biases start at 0 because the network learns them quickly.

### A convenience constructor

```
/**
 * Simple constructor: linear input, sigmoid everywhere else,
 * mean-squared error, plain SGD.
 *
 * @param topology     layer sizes
 * @param learningRate step size for SGD
 */
public NeuralNetwork(int[] topology, double learningRate) {
    this(topology, repeatedSigmoid(topology),
         LossFunction.meanSquaredError(),
         new SGD(learningRate), 42L);
}
```

## 1.4 Putting it together: a 2-8-1 network

```
NeuralNetwork net = new NeuralNetwork(
    new int[]{2, 8, 1},
    new ActivationFunction[]{
        ActivationFunction.linear(),   // input layer
        ActivationFunction.tanh(),     // hidden layer
        ActivationFunction.sigmoid()   // output layer
    },
    LossFunction.meanSquaredError(),
    new Adam(0.01),
    42L
);
```

This builds: an input layer of 2 value-only neurons, a hidden layer of 8 tanh neurons each with 2 weights, and an output layer of 1 sigmoid neuron with 8 weights. Total trainable parameters: `(2+1)×8 + (8+1)×1 = 33`.

---

[← Start here](index.md) · [02 · Forward pass →](02-forward-pass.md)
