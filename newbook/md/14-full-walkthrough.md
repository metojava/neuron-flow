# Complete listing of the core classes

*This chapter reproduces the essential classes in full so you can read them side by side with the explanations from earlier chapters. All code uses `float` precision and lives in the packages shown in chapter 13.*

## 14.1 Neuron

```
package layers;

import java.util.Random;

/**
 * One artificial neuron: holds its weights (one per input),
 * its bias, the activation from the last forward pass, and the
 * error signal (gradient) from the last backward pass.
 */
public class Neuron {

    /** One weight per upstream neuron (input layer: length 0). */
    public float[] weights;

    /** Additive bias applied before activation. */
    public float bias;

    /** Activation a = f(z) cached from the last forward pass. */
    public float value;

    /** Backpropagated error signal for this neuron. */
    public float gradient;

    public Neuron(int numInputs, Random rand) {
        this.weights = new float[numInputs];
        for (int i = 0; i < numInputs; i++) {
            this.weights[i] = rand.nextFloat() * 2.0f - 1.0f;
        }
        this.bias = rand.nextFloat() * 2.0f - 1.0f;
    }
}
```

## 14.2 Layer

```
package layers;

import java.util.Random;

/**
 * A layer is a fixed-size row of neurons that all share the
 * same number of inputs. The input layer is created with zero
 * inputs per neuron: its neurons only store values.
 */
public class Layer {

    /** Neurons of this layer, in order. */
    public Neuron[] neurons;

    public Layer(int numNeurons, int numInputsPerNeuron,
                 Random rand) {
        this.neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            this.neurons[i] =
                new Neuron(numInputsPerNeuron, rand);
        }
    }
}
```

## 14.3 NeuralNetwork (forward + backprop, SIMD-accelerated)

```
package layers;

import activations.ActivationFunction;
import initialization.WeightInitializer;
import losses.LossFunction;
import optim.Optimizer;
import optim.SGD;
import training.Trainer;
import vector.VectorMath;

import java.util.Random;

public class NeuralNetwork {

    private final Layer[] layers;
    private final ActivationFunction[] activations;
    private final LossFunction lossFunction;
    private final Optimizer optimizer;
    private final float[] prevBuffer;   // SIMD dot-product buffer
    private final float[] deltaBuffer;  // backprop delta buffer

    public NeuralNetwork(
            int[] topology,
            ActivationFunction[] activations,
            LossFunction lossFunction,
            Optimizer optimizer,
            long seed,
            WeightInitializer initializer) {
        // ... validation omitted for brevity ...
        this.activations = activations.clone();
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.layers = new Layer[topology.length];

        int maxSize = 0;
        for (int size : topology) {
            if (size > maxSize) maxSize = size;
        }
        this.prevBuffer = new float[maxSize];
        this.deltaBuffer = new float[maxSize];

        Random rand = new Random(seed);
        layers[0] = new Layer(topology[0], 0, rand);

        for (int l = 1; l < topology.length; l++) {
            int fanIn = topology[l - 1];
            int fanOut = topology[l];
            layers[l] = new Layer(topology[l], fanIn, rand);
            for (Neuron neuron : layers[l].neurons) {
                initializer.initialize(
                    neuron.weights, fanIn, fanOut, rand);
                neuron.bias = 0.0f;
            }
        }
    }

    /** SIMD-accelerated forward propagation. */
    public float[] forwardPropagate(float[] inputs) {
        // ... input validation ...
        for (int i = 0; i < inputs.length; i++) {
            layers[0].neurons[i].value = inputs[i];
        }

        for (int l = 1; l < layers.length; l++) {
            Layer previous = layers[l - 1];
            Layer current  = layers[l];
            int prevLen = previous.neurons.length;

            // copy previous activations into a flat buffer
            for (int j = 0; j < prevLen; j++) {
                prevBuffer[j] = previous.neurons[j].value;
            }

            for (Neuron neuron : current.neurons) {
                float sum = neuron.bias
                    + VectorMath.dot(prevBuffer,
                                     neuron.weights, prevLen);
                neuron.value = activations[l].activate(sum);
            }
        }

        Layer output = layers[layers.length - 1];
        float[] result = new float[output.neurons.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = output.neurons[i].value;
        }
        return result;
    }

    /** One mini-batch: forward, backprop, accumulate, step. */
    public void trainBatch(float[][] inputs,
                           float[][] targets) {
        // ... validation ...
        float[] parameters = flattenParameters();
        float[] gradients = new float[parameters.length];

        for (int sample = 0; sample < inputs.length; sample++) {
            float[] prediction =
                forwardPropagate(inputs[sample]);

            float[] outputGradient =
                lossFunction.gradient(prediction, targets[sample]);

            Layer outputLayer = layers[layers.length - 1];
            for (int i = 0; i < outputLayer.neurons.length; i++) {
                Neuron neuron = outputLayer.neurons[i];
                neuron.gradient = outputGradient[i]
                    * activations[layers.length - 1]
                        .derivativeFromOutput(neuron.value);
            }

            // hidden-layer deltas, SIMD-accumulated
            for (int l = layers.length - 2; l > 0; l--) {
                Layer current = layers[l];
                Layer next    = layers[l + 1];
                int curLen = current.neurons.length;

                for (int i = 0; i < curLen; i++) {
                    deltaBuffer[i] = 0.0f;
                }
                for (Neuron nextNeuron : next.neurons) {
                    VectorMath.addScaledScalar(
                        deltaBuffer, 0, nextNeuron.gradient,
                        nextNeuron.weights, curLen);
                }
                for (int i = 0; i < curLen; i++) {
                    Neuron neuron = current.neurons[i];
                    neuron.gradient = deltaBuffer[i]
                        * activations[l].derivativeFromOutput(
                            neuron.value);
                }
            }

            // accumulate parameter gradients, SIMD-scattered
            int k = 0;
            for (int l = 1; l < layers.length; l++) {
                Layer previous = layers[l - 1];
                int prevLen = previous.neurons.length;
                for (int j = 0; j < prevLen; j++) {
                    prevBuffer[j] = previous.neurons[j].value;
                }
                for (Neuron neuron : layers[l].neurons) {
                    VectorMath.addScaledScalar(
                        gradients, k, neuron.gradient,
                        prevBuffer, prevLen);
                    k += prevLen;
                    gradients[k++] += neuron.gradient;
                }
            }
        }

        for (int i = 0; i < gradients.length; i++) {
            gradients[i] /= inputs.length;
        }
        optimizer.update(parameters, gradients);
        unflattenParameters(parameters);
    }

    // fit(), calculateLoss(), flatten/unflatten omitted —
    // see the source file for the full implementations.
}
```

## 14.4 VectorMath (SIMD helpers)

```
package vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class VectorMath {

    private static final VectorSpecies<Float> SPECIES =
        FloatVector.SPECIES_PREFERRED;

    /** Dot product of the first n elements of a and b. */
    public static float dot(float[] a, float[] b, int n) {
        int i = 0;
        int limit = SPECIES.loopBound(n);
        FloatVector acc = FloatVector.zero(SPECIES);

        for (; i < limit; i += SPECIES.length()) {
            FloatVector va =
                FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb =
                FloatVector.fromArray(SPECIES, b, i);
            acc = va.fma(vb, acc);          // acc += va * vb
        }

        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /** out[outOffset + i] += scalar * b[i] for i in [0, n). */
    public static void addScaledScalar(float[] out, int outOffset,
                                       float scalar, float[] b,
                                       int n) {
        int i = 0;
        int limit = SPECIES.loopBound(n);
        FloatVector vScalar =
            FloatVector.broadcast(SPECIES, scalar);

        for (; i < limit; i += SPECIES.length()) {
            FloatVector vo = FloatVector.fromArray(
                SPECIES, out, outOffset + i);
            FloatVector vb =
                FloatVector.fromArray(SPECIES, b, i);
            vo = vScalar.fma(vb, vo);
            vo.intoArray(out, outOffset + i);
        }

        for (; i < n; i++) {
            out[outOffset + i] += scalar * b[i];
        }
    }

    // relu(float[]) and addScalar(float[], float) also available.

    private VectorMath() { }
}
```

## 14.5 Trainer

```
package training;

import java.util.Random;

public class Trainer {

    private final int epochs;
    private final int batchSize;
    private final boolean shuffle;
    private final Random random;

    public Trainer(int epochs, int batchSize, boolean shuffle,
                   long seed) {
        if (epochs <= 0 || batchSize <= 0) {
            throw new IllegalArgumentException(
                "Epochs and batch size must be positive");
        }
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.random = new Random(seed);
    }

    public int getEpochs()        { return epochs; }
    public int getBatchSize()     { return batchSize; }
    public boolean shouldShuffle(){ return shuffle; }
    public Random getRandom()     { return random; }
}
```

## 14.6 LossFunction (softmax cross-entropy excerpt)

```
package losses;

public interface LossFunction {

    float loss(float[] predictions, float[] targets);
    float[] gradient(float[] predictions, float[] targets);

    /**
     * Numerically stable softmax cross-entropy. The gradient
     * is (softmax(logits) - target); pair with a linear output.
     */
    static LossFunction softmaxCrossEntropy() {
        return new LossFunction() {
            @Override
            public float loss(float[] p, float[] t) {
                float max = p[0];
                for (int i = 1; i < p.length; i++) {
                    if (p[i] > max) max = p[i];
                }
                float sumExp = 0.0f;
                for (int i = 0; i < p.length; i++) {
                    sumExp += (float) Math.exp(p[i] - max);
                }
                float logSumExp =
                    max + (float) Math.log(sumExp);
                float nll = 0.0f;
                for (int i = 0; i < p.length; i++) {
                    nll += t[i] * (logSumExp - p[i]);
                }
                return nll;
            }

            @Override
            public float[] gradient(float[] p, float[] t) {
                float[] g = new float[p.length];
                float max = p[0];
                for (int i = 1; i < p.length; i++) {
                    if (p[i] > max) max = p[i];
                }
                float sumExp = 0.0f;
                for (int i = 0; i < p.length; i++) {
                    float e = (float) Math.exp(p[i] - max);
                    g[i] = e;
                    sumExp += e;
                }
                for (int i = 0; i < p.length; i++) {
                    g[i] = g[i] / sumExp - t[i];
                }
                return g;
            }
        };
    }
}
```

> **The remaining classes** — `ActivationFunction`, `SGD`, `Adam`, `WeightInitializer`, `ClassificationMetrics`, `MNISTLoader`, `FashionMNISTLoader`, and the runners — follow the same pattern: small, self-contained, fully documented in the source tree under `src/main/java/`.

---

[← 13 · Run it](13-run.md) · [15 · Practice →](15-practice.md)
