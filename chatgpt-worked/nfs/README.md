# Java deep learning library

A small dense feed-forward neural network for learning and experimentation. Java 17 is required.

## Run examples

```powershell
mvn compile
mvn exec:java
java -cp target/classes org.example.deeplearning.examples.RegressionExample
java -cp target/classes org.example.deeplearning.examples.MNISTRunner --quick
```

MNIST downloads its dataset on first run. The regression example fits `y = 2x + 1` with a linear output, Huber loss and RMSProp.

## Components

- `core`: `NeuralNetwork`, `Layer`, `Neuron`.
- `activation`: scalar activation functions.
- `loss`: mean squared error and binary cross-entropy factories, plus `MeanAbsoluteError` and `HuberLoss`.
- `optim`: `SGD`, `Adam`, and `RMSProp`. Create a separate optimizer for each network because optimizers hold state.
- `initialization`: `WeightInitializer.xavierUniform()` and `WeightInitializer.heNormal()`. Pass an initializer as the final argument to the six-argument network constructor; existing constructors use Xavier uniform.
- `training`: training configuration.
- `evaluation`: `ClassificationMetrics` for argmax, accuracy, and confusion matrices (rows actual, columns predicted).
- `data` and `examples`: MNIST utilities and runnable programs.

```java
NeuralNetwork network = new NeuralNetwork(
    new int[]{1, 8, 1},
    new ActivationFunction[]{ActivationFunction.linear(), ActivationFunction.relu(), ActivationFunction.linear()},
    new HuberLoss(1.0), new RMSProp(0.01), 42L,
    WeightInitializer.heNormal());
```

Import the corresponding classes from `org.example.deeplearning` subpackages. Outputs and loss gradients are averaged over output dimensions; batch gradients are averaged over examples.

## Verification without Maven

From the project directory in PowerShell:

```powershell
New-Item -ItemType Directory -Force build-check | Out-Null
$sources = Get-ChildItem src/main/java,src/test/java -Recurse -Filter *.java | ForEach-Object { $_.FullName }
javac -d build-check $sources
java -cp build-check org.example.deeplearning.ExpansionChecks
```

This executable check covers numerical loss gradients, RMSProp updates, initialization, metrics, and a small regression training run. It is a standalone program, not a JUnit test.

The library currently uses elementwise activations; softmax and categorical cross-entropy are not yet supported.
