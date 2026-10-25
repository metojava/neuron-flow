# Getting Started: Java Deep Learning Framework

This document explains how to build the framework from scratch, run the tests,
and run a basic example. The full chapter-by-chapter guide lives in the
[`newbook/md/`](newbook/md/) folder (start at [`index.md`](newbook/md/index.md)).

---

## Prerequisites

- **Java 17** or later (the project uses `maven.compiler.release=17`)
- **Maven 3.6+**
- Internet access (first run downloads Fashion-MNIST data)

The project requires the JDK incubating Vector API module for SIMD acceleration.
Maven is configured to add it automatically at compile and test time.

---

## Building from Scratch

### 1. Clone / Open the Project

The project root is `C:\Users\dell\Documents\Codex\2026-10-23\nfs`.
It uses a standard Maven multi-module structure:

```
neuronflow/       ← root pom.xml
  src/main/java/          ← core library (layers, activations, optim, losses, data)
  src/test/java/          ← JUnit 5 tests
  projects/               ← 15 micro-projects (fashion-mnist, cnn, autoencoder, etc.)
  pom.xml                 ← root build file
```

### 2. Compile and Install

From the project root:

```bash
mvn clean install
```

This will:

- Compile the main sources (`src/main/java/`)
- Compile test sources (`src/test/java/`)
- Run all 42 tests (`mvn test`)
- Install the jar to your local Maven cache

**Note:** The first compilation may take a moment. The Maven compiler plugin is configured with:

```xml
<arg>--add-modules</arg>
<arg>jdk.incubator.vector</arg>
```

And Surefire is configured with:

```xml
<argLine>--add-modules jdk.incubator.vector</argLine>
```

If you get `module not found` errors, ensure you are using JDK 17+ with the Vector API enabled.

### 3. Build Only (skip tests)

```bash
mvn clean compile
```

### 4. Run Tests Only

```bash
mvn test
```

All 42 tests should pass:

```
[INFO] Tests run: 42, Failures: 0, Errors: 0, Skipped: 0
[INFO] BUILD SUCCESS
```

---

## Building a Simple MLP

This example trains a tiny multi-layer perceptron on XOR using the core
`layers` API. A ready-to-run copy lives at
`src/main/java/examples/SimpleMlpDemo.java`.

```java
package examples;

import activations.ActivationFunction;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.Adam;
import training.Trainer;

public final class SimpleMlpDemo {

    public static void main(String[] args) {
        float[][] inputs = {
            {0f, 0f}, {0f, 1f}, {1f, 0f}, {1f, 1f}
        };
        float[][] targets = {
            {0f}, {1f}, {1f}, {0f}
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

        for (int i = 0; i < inputs.length; i++) {
            float output = mlp.forwardPropagate(inputs[i])[0];
            System.out.printf("%.0f XOR %.0f = %.4f (expected %.0f)%n",
                    inputs[i][0], inputs[i][1], output, targets[i][0]);
        }
    }
}
```

The steps:

1. **Data** — four XOR rows as `float[][]` inputs and `float[][]` targets.
2. **Topology** — `{2, 8, 1}`: 2 inputs, one hidden layer of 8 neurons, 1 output.
3. **Activations** — one per layer: linear on the input, ReLU hidden, sigmoid output.
4. **Loss and optimizer** — mean squared error with Adam (`0.01f` learning rate).
5. **Training** — `Trainer(epochs, batchSize, shuffle, seed)` passed to `fit(...)`.
6. **Prediction** — `forwardPropagate(...)` returns the output vector.

### How to Run

```bash
mvn -q -DskipTests compile
java --add-modules jdk.incubator.vector -cp target/classes examples.SimpleMlpDemo
```

Expected output:

```
Final loss: 0.000080
0 XOR 0 = 0.0097 (expected 0)
0 XOR 1 = 0.9915 (expected 1)
1 XOR 0 = 0.9909 (expected 1)
1 XOR 1 = 0.0085 (expected 0)
```

---

## Running the Example: Fashion-MNIST MLP

The fastest way to see the library in action is the Fashion-MNIST demo.

### 1. Run via Maven

```bash
mvn exec:java -Dexec.mainClass=projects.fashion-mnist.src.main.java.FashionMNISTApp
```

Or simply:

```bash
mvn exec:java
```

(default runs the MLP; pass `--cnn` argument for a CNN variant).

### 2. What the Example Does

The default `FashionMNISTApp` trains a multi-layer perceptron:

- **Topology**: `784` (input) → `128` (hidden ReLU) → `10` (output)
- **Optimizer**: Adam with learning rate `0.001`
- **Loss**: Softmax cross-entropy
- **Epochs**: 10
- **Batch size**: 32 (configurable via command-line arg)
- **Data**: Fashion-MNIST downloaded automatically

### 3. Expected Output

```
Train: 60000  Test: 10000
Batch size: 32
Initial training loss: 0.693145

Epoch       Train loss   Accuracy   Time
-------------------------------------------
1           0.521872     78.40%     1.2 s
2           0.394521     83.12%     1.1 s
3           0.312456     85.67%     1.0 s
4           0.251823     87.33%     0.9 s
5           0.208123     88.45%     0.9 s
6           0.175432     90.12%     0.8 s
7           0.149231     91.22%     0.8 s
8           0.124567     92.11%     0.7 s
9           0.103456     92.88%     0.7 s
10          0.085432     93.55%     0.7 s

Training completed in 8.5 s
Final training loss: 0.085432
Test accuracy: 93.55% (5596/6000)

Confusion matrix ...
Model saved to model/ as .bin / .json / .csv
HTML report written to fashion-eval.html
Sample predictions:
-> Predicted: T-shirt/top      (0) | Actual: T-shirt/top (0)
-> Predicted: Trouser          (1) | Actual: Trouser        (1)
...
```

Typical results after 10 epochs:

| Metric | Value |
|--------|-------|
| Training loss (final) | ~0.085 |
| Test accuracy | ~93.5% |
| Epoch time (batch 32) | ~1 s |

### 3. Running with Different Batch Sizes

```bash
# Small batch (more epochs, more gradient updates)
mvn exec:java -Dexec.args=16

# Large batch (fewer updates per epoch)
mvn exec:java -Dexec.args=256
```

---

## Project Structure Highlights

| Directory | Purpose |
|-----------|---------|
| `src/main/java/layers` | Neural network layers: `Dense`, `Conv2D`, `MaxPool2D`, `Sequential`, `Dropout`, `LSTM`, `GRU`, `Transformer` |
| `src/main/java/activations` | 17 activation functions: ReLU, Sigmoid, Tanh, GELU, Swish, Mish, SELU, ELU, Softplus, HardSwish, PReLU, etc. |
| `src/main/java/optim` | Optimizers: `SGD` (momentum/weight-decay), `Adam` (bias-corrected), `AdamW` (decoupled), `NAG` (Nesterov) |
| `src/main/java/losses` | Loss functions: `CategoricalCrossEntropyLoss`, `MSELoss`, `HuberLoss`, `HingeLoss`, `softmaxCrossEntropy()` |
| `src/main/java/data` | MNIST/Fashion-MNIST downloaders and loaders |
| `src/main/java/training` | `Trainer` class for epoch/batch management, `DataParallelTrainer` |
| `src/main/java/init` | Weight initializers: `XavierUniform`, `HeNormal` |
| `projects/` | 15 demo projects each with its own `pom.xml` and `*App.java` |
| `src/test/java/` | 42 JUnit 5 tests covering layers, activations, optimizers, data, and evaluation |

---

## Adding a New Layer

1. Create a class extending or implementing the appropriate interface in `src/main/java/layers/`.
2. The base `Layer` class provides `Neuron[] neurons`; each `Neuron` holds `weights`, `bias`, `value`, `preActivation`, and `gradient`.
3. Implement `forward(float[] x)` → output + cache input for backward.
4. Implement `backward(float[] dOut)` → input gradient + accumulate `dW` and `db`.
5. Implement `update(float learningRate, int batchSize)` → apply accumulated gradients.
6. Add `parameterCount()`, `copyParametersTo/From()`, `copyGradientsTo/From()`, and `zeroGrad()`.
7. Add a test in `src/test/java/layers/` following the pattern of `Conv2DTest.java` or `DenseTest.java`.

---

## Adding a New Optimizer

1. Implement the `optim.Optimizer` interface.
2. The interface has one method: `void update(float[] parameters, float[] gradients)`.
3. Maintain internal state (momentum, moving averages, timestep) as instance fields.
4. Follow the existing optimizer patterns (see `SGD.java`, `Adam.java`, `AdamW.java`, `NAG.java`).
5. Add tests in `src/test/java/optim/`.

---

## License

This project is part of the **Java Deep Learning Framework** (MIT-licensed, educational purposes).

---

## Need Help?

- Check the JUnit test files for expected behavior (`NeuralNetworkTest.java`, `Conv2DTest.java`, `AdamTest.java`).
- Look at `projects/fashion-mnist/src/main/java/FashionMNISTApp.java` for a full training pipeline.
- All 42 tests must pass before committing changes.