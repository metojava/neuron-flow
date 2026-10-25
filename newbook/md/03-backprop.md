# Gradients from the error backwards

*After the forward pass, the network knows how wrong it was. Backpropagation walks that error from the output layer back to the first hidden layer using the chain rule, producing one gradient per weight.*

## 3.1 The chain rule in one picture

**Loss** (how wrong) → **∂L/∂output** (error signal) → **× f′(z)** (activation slope) → **δ** (per neuron) → **× input** (per weight)

Output layer: δ = (∂L/∂a) · f′(z)

 Hidden layer: δⱼ = ( Σᵢ δᵢ · wᵢⱼ ) · f′(zⱼ)

 Weight grad: ∂L/∂wⱼₖ = δⱼ · aₖ

 Bias grad: ∂L/∂bⱼ = δⱼ

## 3.2 The method: `trainBatch`

```
/**
 * Trains on one mini-batch. Gradients are accumulated over every
 * sample first, averaged, and only then handed to the optimizer.
 *
 * @param inputs  batch of examples, [batchSize][inputSize]
 * @param targets batch of targets,  [batchSize][outputSize]
 * @throws IllegalArgumentException if any dimension is inconsistent
 */
public void trainBatch(double[][] inputs, double[][] targets) {
    validateDataset(inputs, targets);

    int parameterCount = parameterCount();
    double[] parameters = flattenParameters();
    double[] gradients  = new double[parameterCount];

    for (int sample = 0; sample < inputs.length; sample++) {

        // --- forward ---
        double[] prediction = forwardPropagate(inputs[sample]);

        if (targets[sample].length != prediction.length) {
            throw new IllegalArgumentException(
                "Target dimension does not match output");
        }

        // --- 1. output-layer gradients ---
        double[] outputGradient =
            lossFunction.gradient(prediction, targets[sample]);

        Layer outputLayer = layers[layers.length - 1];

        for (int i = 0; i < outputLayer.neurons.length; i++) {
            Neuron neuron = outputLayer.neurons[i];
            neuron.gradient = outputGradient[i]
                * activations[layers.length - 1]
                    .derivativeFromOutput(neuron.value);
        }

        // --- 2. hidden layers, walking backwards ---
        for (int l = layers.length - 2; l > 0; l--) {
            Layer current = layers[l];
            Layer next    = layers[l + 1];

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

        // --- 3. parameter gradients, same order as flatten ---
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

    // Average over the batch, then update.
    for (int i = 0; i < gradients.length; i++) {
        gradients[i] /= inputs.length;
    }

    optimizer.update(parameters, gradients);
    unflattenParameters(parameters);
}
```

### Step by step

1. **Forward** the sample so every neuron caches its activation.
2. **Output δ:** multiply the loss gradient by the output activation's derivative — this is `∂L/∂z`.
3. **Hidden δ:** for each hidden neuron, sum (next layer's δ × connecting weight), then multiply by its own derivative. The loop runs from the second-to-last layer down to layer 1 — the input layer is skipped because it has no weights.
4. **Parameter gradients:** δ × upstream activation for weights, plain δ for biases. The order matches `flattenParameters()` exactly so indices line up.
5. **Average** the accumulated gradients over the batch size.
6. **Update:** give the flattened parameters and gradients to the optimizer, then write the new values back into the neurons.

### Why the trick `derivativeFromOutput` works

Most activation derivatives can be written using the *output* instead of the input:

- Sigmoid: `f′(z) = a(1−a)` where `a = σ(z)`
- Tanh: `f′(z) = 1 − a²`
- ReLU: `1 if a > 0 else 0`

Since `neuron.value` already holds `a`, no second pass is needed.

### Helper methods used above

| Method | Return | Logic |
| --- | --- | --- |
| `parameterCount()` | `int` | Sums `weights.length + 1` (the +1 is the bias) over every neuron in layers 1..n. |
| `flattenParameters()` | `double[]` | Copies every weight and bias, layer by layer, neuron by neuron, into one long array. |
| `unflattenParameters(p)` | `void` | Writes the (updated) flat array back into the neurons in the same order. |
| `validateDataset(x, y)` | `void` | Checks row counts match and each row matches input/output sizes. |

> **Averaging convention.** Gradients are divided by the batch size *and* the loss gradient already divides by the output size. This keeps step magnitudes independent of batch and class count — useful when comparing runs.

---

[← 02 · Forward pass](02-forward-pass.md) · [04 · Activations →](04-activations.md)
