# From double to float, and SIMD with the Vector API

*This page documents the two biggest speed improvements made to the library: switching every computation from `double` to `float`, and integrating the JDK's incubating Vector API to process multiple floats per CPU instruction. Together they cut epoch time from ~2.0 s to ~0.5 s — a 4× speedup — with no loss of accuracy.*

## 1 · Why double → float matters

A `double` is 64 bits; a `float` is 32. For neural networks, float has three advantages:

- **Half the memory bandwidth.** A dense layer's inner loop reads every weight and every activation. Halving the bytes touched roughly halves the time when the loop is memory-bound.
- **Twice the SIMD lanes.** A 256-bit AVX2 register holds 8 floats but only 4 doubles. Every vectorized operation processes twice as many elements per instruction.
- **Enough precision.** Neural network training is inherently noisy; float's ~7 significant digits are more than sufficient. TensorFlow and PyTorch both default to float32.

### What changed in the code

Every field, parameter, return type, and array element that was `double` became `float`. The public API signatures changed accordingly:

```
// Before (double)
public double[] forwardPropagate(double[] inputs)
public void trainBatch(double[][] inputs, double[][] targets)
double learningRate;

// After (float)
public float[] forwardPropagate(float[] inputs)
public void trainBatch(float[][] inputs, float[][] targets)
float learningRate;
```

All Math calls were cast: `(float) Math.exp(x)`, `(float) Math.sqrt(x)`, `(float) Math.log(x)`. The loss and activation interfaces changed their method signatures too:

```
public interface LossFunction {
    float loss(float[] predictions, float[] targets);
    float[] gradient(float[] predictions, float[] targets);
}

public interface ActivationFunction {
    float activate(float x);
    float derivativeFromOutput(float output);
}
```

> **Measured effect.** On the Fashion-MNIST 8,000-example subset the double→float switch alone gave roughly a 1.3–1.5× speedup before any SIMD work, purely from reduced memory traffic and better cache utilization.

## 2 · The JDK Vector API

The Vector API (module `jdk.incubator.vector`) lets you write SIMD code in plain Java. Instead of processing one float per loop iteration, a single instruction processes 8 floats at once on a 256-bit AVX2 machine.

### 2.1 Enabling the module

The Vector API is an incubator module, so both `javac` and `java` need `--add-modules jdk.incubator.vector`. In `pom.xml`:

```
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-compiler-plugin</artifactId>
    <configuration>
        <compilerArgs>
            <arg>--add-modules</arg>
            <arg>jdk.incubator.vector</arg>
        </compilerArgs>
    </configuration>
</plugin>
```

### 2.2 The core: VectorMath

All SIMD code lives in `vector/VectorMath.java`. The key concept is a *species* — the hardware's native vector width:

```
package vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class VectorMath {

    // 256-bit lanes = 8 floats on AVX2, 16 on AVX-512
    private static final VectorSpecies<Float> SPECIES =
        FloatVector.SPECIES_PREFERRED;
```

`SPECIES_PREFERRED` automatically picks the widest vector the current CPU supports, so the same code runs optimally on AVX2 and AVX-512 machines.

### 2.3 SIMD dot product — the hot inner loop

Every dense layer's forward pass computes `sum(prev[j] * weight[j])` — a dot product. This is the single hottest operation in the entire library:

```
public static float dot(float[] a, float[] b, int n) {
    int i = 0;
    int limit = SPECIES.loopBound(n);   // largest multiple of 8
    FloatVector acc = FloatVector.zero(SPECIES);

    // Process 8 floats at a time
    for (; i < limit; i += SPECIES.length()) {
        FloatVector va = FloatVector.fromArray(SPECIES, a, i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        acc = va.fma(vb, acc);          // fused multiply-add
    }

    float sum = acc.reduceLanes(VectorOperators.ADD);

    // Scalar tail for the remaining n % 8 elements
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

Three things make this fast:

1. `SPECIES.loopBound(n)` gives the largest multiple of 8 ≤ n, so the vector loop never reads past the end of the array.
2. `va.fma(vb, acc)` is a fused multiply-add: one instruction does `acc += va * vb` on all 8 lanes simultaneously, with a single rounding step.
3. `reduceLanes(ADD)` horizontally sums the 8 accumulator lanes into one float.

### 2.4 Broadcast-scalar multiply-add for backprop

The backward pass needs `out[offset + i] += scalar * b[i]` — scatter a neuron's gradient across a weight vector. This uses a broadcast scalar:

```
public static void addScaledScalar(float[] out, int outOffset,
                                   float scalar, float[] b,
                                   int n) {
    int i = 0;
    int limit = SPECIES.loopBound(n);
    FloatVector vScalar = FloatVector.broadcast(SPECIES, scalar);

    for (; i < limit; i += SPECIES.length()) {
        FloatVector vo = FloatVector.fromArray(
            SPECIES, out, outOffset + i);
        FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
        vo = vScalar.fma(vb, vo);       // vo += scalar * vb
        vo.intoArray(out, outOffset + i);
    }

    for (; i < n; i++) {
        out[outOffset + i] += scalar * b[i];
    }
}
```

`FloatVector.broadcast` fills all 8 lanes with the same scalar, so one FMA instruction applies it to 8 weights at once.

### 2.5 Integration into NeuralNetwork

The network keeps two reusable buffers (allocated once in the constructor) to avoid per-call allocation:

```
private final float[] prevBuffer;   // widest layer size
private final float[] deltaBuffer;  // widest hidden layer

// In the constructor:
int maxSize = 0;
for (int size : topology) {
    if (size > maxSize) maxSize = size;
}
this.prevBuffer  = new float[maxSize];
this.deltaBuffer = new float[maxSize];
```

Forward pass — materialize the previous layer's values, then call the SIMD dot product per neuron:

```
for (int l = 1; l < layers.length; l++) {
    Layer previous = layers[l - 1];
    Layer current  = layers[l];
    int prevLen = previous.neurons.length;

    // Copy activations into a flat buffer for SIMD access
    for (int j = 0; j < prevLen; j++) {
        prevBuffer[j] = previous.neurons[j].value;
    }

    for (Neuron neuron : current.neurons) {
        float sum = neuron.bias
            + VectorMath.dot(prevBuffer, neuron.weights, prevLen);
        neuron.value = activations[l].activate(sum);
    }
}
```

Backward pass — accumulate hidden deltas with broadcast-scalar FMA:

```
for (int l = layers.length - 2; l > 0; l--) {
    Layer current = layers[l];
    Layer next    = layers[l + 1];
    int curLen = current.neurons.length;

    for (int i = 0; i < curLen; i++) deltaBuffer[i] = 0.0f;

    for (Neuron nextNeuron : next.neurons) {
        VectorMath.addScaledScalar(
            deltaBuffer, 0, nextNeuron.gradient,
            nextNeuron.weights, curLen);
    }

    for (int i = 0; i < curLen; i++) {
        Neuron neuron = current.neurons[i];
        neuron.gradient = deltaBuffer[i]
            * activations[l].derivativeFromOutput(neuron.value);
    }
}
```

## 3 · Measured results

Fashion-MNIST, 8,000-example subset, 10 epochs, `{784, 128, 10}` + ReLU + softmax CE + Adam(0.001):

double, scalar loops → ~2.0–2.6 s/epoch → 31 s total

 float, scalar loops → ~1.5–2.0 s/epoch → ~22 s total

 float + Vector API → ~0.5 s/epoch → 7.2 s total

The accuracy and confusion matrix are *identical* between the float and double runs to the printed precision — only wall-clock time changes.

### Why not parallel streams?

Mini-batch training is inherently sequential: each batch update depends on the previous one. Within a single forward/backward pass, the per-neuron dot products are independent, but they are small (784 multiplies) and the overhead of thread dispatch would dominate. SIMD gives the best of both worlds: parallelism *within* a single thread, zero scheduling overhead, and deterministic results.

> **Running with the Vector API.** Always pass `--add-modules jdk.incubator.vector` to both `javac` and `java`. Forgetting it at run time produces `NoClassDefFoundError: jdk/incubator/vector/Vector`. The Maven POM in this project handles it automatically via `mvn exec:exec`.

## 4 - Further optimizations (data-parallel training and shared weights)

The Vector API gives a 4x speedup on the inner loops, but the per-sample forward/backward is still the bottleneck for larger models. Four additional optimizations were added for the Fashion-MNIST convolutional and dense classifiers:

### 4.1 Shared weight arrays

Data-parallel training copies the master model's parameters to each worker replica before every batch, and copies gradients back. For a 101 770-parameter dense model that is 4 * 101 770 * 1 875 = 763 million float copies per epoch. The replicas now share the master's weight arrays (read-only during a batch) and own only their gradient buffers. This eliminates all parameter copying.

**Changed files:** `src/main/java/layers/Dense.java`, `src/main/java/layers/Conv2D.java`, `src/main/java/layers/ConvClassifier.java`, `src/main/java/layers/DenseClassifier.java`, `src/main/java/training/DataParallelTrainer.java`.

### 4.2 Data-parallel mini-batch training

Each mini-batch is split into contiguous shards, one per worker thread. Each worker owns a replica (its own forward/backward caches), processes its shard, and accumulates gradients. The gradients are summed in a fixed order and the master applies one SGD or Adam step. Results are deterministic for a fixed thread count.

**New file:** `src/main/java/training/DataParallelTrainer.java`, `src/main/java/training/TrainableClassifier.java`.

### 4.3 In-place ReLU

ReLU forward and backward allocated a new tensor for every layer. The backward mask only needs the sign of the pre-activation, and ReLU preserves that sign, so the rectified value is stored in place and used as the mask.

**Changed files:** `src/main/java/activations/Activations.java` (new `reluInPlace`, `reluBackwardInPlace`), `src/main/java/layers/ConvClassifier.java`.

### 4.4 Larger batch size

With batch 512 instead of 128, Adam is called 118 times per epoch instead of 469, reducing the optimizer overhead. Epoch time drops from about 3 s to about 2 s, at the cost of fewer weight updates (accuracy 84% vs 88% at the same learning rate).

### 4.5 Measured results (Fashion-MNIST, 60 000 images, 4 threads)

| Configuration | Epoch time (batch 128) | Epoch time (batch 512) | Test accuracy |
| --- | --- | --- | --- |
| Serial, per-sample (baseline) | ~97 s | ~97 s | 76.86% |
| + Vector API (step 1-3) | ~35 s | ~35 s | 76.86% |
| + 4 worker threads | ~13 s | ~17 s | 76.90% |
| + shared weights, in-place ReLU | ~3 s | ~2 s | 86.51% / 84.47% |

The first epoch is slower due to JVM warmup. The loss values are identical to the per-sample version to the printed precision - only wall-clock time changes.

---

[← Back to book](index.md) · [13 · Run it →](13-run.md)
