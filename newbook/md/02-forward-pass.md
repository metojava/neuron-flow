# How an input becomes a prediction

*The forward pass computes, layer by layer, `z = w·x + b` followed by `a = f(z)`. We read `forwardPropagate` and walk through the arithmetic by hand.*

## 2.1 The method

```
/**
 * Runs one input vector through the whole network.
 *
 * Each neuron in layer l computes
 *     z = bias + Σ (previous.value × weight)
 *     neuron.value = activation[l](z)
 * and the final layer's values are copied out as the result.
 *
 * @param inputs one example, length must equal layer-0 size
 * @return       one output per output-layer neuron
 * @throws IllegalArgumentException if the length is wrong or a value
 *                                  is NaN/infinite
 */
public double[] forwardPropagate(double[] inputs) {
    if (inputs == null ||
        inputs.length != layers[0].neurons.length) {
        throw new IllegalArgumentException(
            "Input dimension does not match network");
    }

    // Step 1: copy the raw input into the input layer.
    for (int i = 0; i < inputs.length; i++) {
        if (!Double.isFinite(inputs[i])) {
            throw new IllegalArgumentException("Inputs must be finite");
        }
        layers[0].neurons[i].value = inputs[i];
    }

    // Step 2: every later layer reads the previous layer's values.
    for (int l = 1; l < layers.length; l++) {
        Layer previous = layers[l - 1];
        Layer current  = layers[l];

        for (Neuron neuron : current.neurons) {
            double sum = neuron.bias;

            for (int j = 0; j < previous.neurons.length; j++) {
                sum += previous.neurons[j].value
                     * neuron.weights[j];
            }

            neuron.value = activations[l].activate(sum);
        }
    }

    // Step 3: copy the output layer's activations into an array.
    Layer output = layers[layers.length - 1];
    double[] result = new double[output.neurons.length];

    for (int i = 0; i < result.length; i++) {
        result[i] = output.neurons[i].value;
    }

    return result;
}

/**
 * Alias for {@link #forwardPropagate(double[])}.
 *
 * @param inputs one example
 * @return network outputs
 */
public double[] predict(double[] inputs) {
    return forwardPropagate(inputs);
}
```

### Parameters in plain language

| Parameter | Shape | Role |
| --- | --- | --- |
| `inputs` | `[inputSize]` | One sample — 784 pixel values in [0,1] for Fashion MNIST. |
| return `result` | `[outputSize]` | One activation per class — 10 numbers for the classifier. |

## 2.2 Worked example: 2 → 1 net

Take one neuron with weights `w = [0.5, -0.3]`, bias `b = 0.1`, tanh activation, and input `x = [1.0, 2.0]`.

z = 0.1 + (1.0 × 0.5) + (2.0 × −0.3)

 z = 0.1 + 0.5 − 0.6 = 0.0

 a = tanh(0.0) = 0.0

The loop in the code performs exactly this: initialize `sum` with the bias, add each `previous.value × weight`, then apply the activation.

## 2.3 What happens behind the scenes

**Input** (a⁰ = x) → **z¹ = W¹a⁰ + b¹** → **a¹ = f¹(z¹)** → **z² = W²a¹ + b²** → **output** (a²)

Notice that `neuron.value` is overwritten on every pass. That's intentional: forward propagation keeps only the most recent activations, which are precisely the values backpropagation needs (via the derivative-from-output trick in `ActivationFunction`).

> **Input layer has no activation.** The code starts the loop at `l = 1`, so `activations[0]` is never applied to raw inputs. Using `linear()` for index 0 is a convention that keeps the arrays aligned.

## 2.4 Complete runnable example

```
import java.util.Random;

public class ForwardDemo {
    public static void main(String[] args) {
        // Network: 2 inputs, 1 hidden tanh neuron, 1 sigmoid output.
        NeuralNetwork net = new NeuralNetwork(
            new int[]{2, 1, 1},
            new ActivationFunction[]{
                ActivationFunction.linear(),
                ActivationFunction.tanh(),
                ActivationFunction.sigmoid()
            },
            LossFunction.meanSquaredError(),
            new SGD(0.1),
            42L
        );

        double[] output = net.predict(new double[]{1.0, 2.0});

        System.out.println("Output: " + output[0]);
        // A value in (0, 1) because the last activation is sigmoid.
    }
}
```

---

[← 01 · Foundations](01-foundations.md) · [03 · Backpropagation →](03-backprop.md)
