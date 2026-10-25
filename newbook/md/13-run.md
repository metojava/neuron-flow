# Compile and run with Maven or the CLI

*This chapter shows how to compile and run the project on your machine. Two paths: Maven (recommended, handles the Vector API module flags for you) or plain `javac`/`java` if you prefer no build tool.*

## 13.1 Requirements

- **JDK 17+** (the project targets `--release 17`; the Vector API incubator module works on 17–23).
- **Maven 3.8+** (optional but convenient).
- An internet connection for the first run — the Fashion-MNIST downloader fetches four gzip files (~30 MB total) from GitHub.

## 13.2 Run with Maven

From the project root (the folder containing `pom.xml`):

```
# compile everything
mvn compile

# train the Fashion-MNIST classifier (downloads data on first run)
mvn exec:exec

# run tests, if any
mvn test
```

`mvn exec:exec` forks a child JVM with the required `--add-modules jdk.incubator.vector` flag (configured in `pom.xml`) and launches `examples.FashionMNISTRunner`. The first run downloads the dataset into `data/fashion-mnist/`; subsequent runs reuse the cached files.

## 13.3 Expected output

```
Loading Fashion-MNIST dataset...
Already available: data\fashion-mnist\train-images-idx3-ubyte.gz
Already available: data\fashion-mnist\train-labels-idx1-ubyte.gz
Already available: data\fashion-mnist\t10k-images-idx3-ubyte.gz
Already available: data\fashion-mnist\t10k-labels-idx1-ubyte.gz
Fashion-MNIST files are ready in: C:\...\data\fashion-mnist
Training examples: 60000
Test examples: 10000
Examples used for training: 8000

Initial training loss: 2.494409

Training (Keras-like simple model)...

Epoch      Train loss    Accuracy     Time
-------------------------------------------
1          0.524989      80.63%       0.7 s
2          0.456892      83.28%       0.5 s
3          0.389585      85.78%       0.5 s
...
10         0.257187      90.50%       0.5 s

Training completed in 7.2 s
Final training loss: 0.257187
Test accuracy: 84.07%

Final evaluation:
Test accuracy: 84.07% (8407/10000)

Confusion matrix (rows = actual, cols = predicted):
                0     1     2     3     4     5     6     7     8     9
T-shirt/top    876     1    18    23     3     0    63     0    16     0
Trouser          9   960     1    22     5     0     1     0     2     0
...
Ankle boot       1     0     0     1     0    11     1   107     2   877

class         precision     recall         f1
T-shirt/top       0.727      0.876      0.795
Trouser           0.982      0.960      0.971
...
```

## 13.4 Run without Maven

Compile every source file, then run with the vector module enabled:

```
# from the project root
javac --add-modules jdk.incubator.vector \
      -d out \
      $(find src/main/java -name "*.java")

java --add-modules jdk.incubator.vector \
     -cp out \
     examples.FashionMNISTRunner
```

On Windows PowerShell, replace `$(find ...)` with an explicit file list or use:

```
Get-ChildItem -Recurse src/main/java/*.java |
  ForEach-Object { $_.FullName } |
  javac --add-modules jdk.incubator.vector -d out @()
```

## 13.5 Why `--add-modules jdk.incubator.vector`?

The Vector API lives in the `jdk.incubator.vector` module, which is not resolved by default. Both `javac` (to compile `VectorMath`) and `java` (to load the classes at runtime) need the flag. Forgetting it at run time produces:

```
java.lang.NoClassDefFoundError: jdk/incubator/vector/Vector
```

> **Speed.** With the Vector API active, each training epoch on the 8,000-example subset takes roughly 0.5 s instead of 2 s — about a 4× speedup on the forward and backward inner loops. The accuracy is identical; only the wall-clock time changes.

## 13.6 Saving and loading models

The network can save its topology + parameters to disk in three formats, chosen by file extension or explicitly via the `ModelFormat` enum:

```
// inferred from extension
net.saveModel(Path.of("model.bin"));    // BINARY
net.saveModel(Path.of("model.json"));   // JSON
net.saveModel(Path.of("model.csv"));    // CSV

// explicit format
net.saveModel(Path.of("model.dat"), ModelFormat.BINARY);

// reload — pass the constructor arguments again
NeuralNetwork net2 = NeuralNetwork.loadModel(
    Path.of("model.json"),
    ModelFormat.JSON,
    activations, lossFunction, optimizer, seed, initializer);
```

- **BINARY** (`.bin`) — compact `DataOutputStream` format. Smallest file, fastest load.
- **JSON** (`.json`) — human-readable, diff-friendly. Larger but easy to inspect or load from other languages.
- **CSV** (`.csv`) — one row per neuron, one column per weight. Opens directly in a spreadsheet.

> **Round-trip verified.** A model saved and reloaded in any of the three formats produces identical predictions to the in-memory original (verified on 500 Fashion-MNIST test images).

## 13.7 HTML evaluation report

After training, `HtmlReport.write()` produces a self-contained HTML file with the confusion matrix, per-class precision/recall/F1, and a gallery of test images grouped by true class — each image embedded as a base64 PNG data URI, so the file needs no external assets:

```
HtmlReport.write(
    Path.of("fashion-eval.html"),
    confusion, FashionMNISTLoader.CLASS_NAMES,
    testImages, testLabels, predicted,
    2  // upscale factor: 28x28 -> 56x56
);
```

## 13.8 Project layout (current)

```
src/main/java/
├── activations/
│   ├── ActivationFunction.java   # sigmoid, tanh, relu, leakyRelu, linear,
│   │                              # elu, selu, swish, gelu, softplus
│   └── Softmax.java
├── data/
│   ├── MNISTDownloader.java      # generic gzip IDX downloader
│   ├── MNISTLoader.java          # generic IDX parser + resolve() + one-hot
│   ├── FashionMNISTDownloader.java
│   └── FashionMNISTLoader.java   # CLASS_NAMES, NUM_CLASSES, Dataset
├── evaluation/
│   ├── ClassificationMetrics.java # argmax, accuracy, confusion, P/R/F1
│   ├── HtmlReport.java            # self-contained HTML report
│   └── ImageEncoder.java          # float[] -> base64 PNG data URI
├── examples/
│   ├── FashionMNISTRunner.java
│   ├── MNISTRunner.java
│   └── Main.java
├── initialization/
│   └── WeightInitializer.java    # xavierUniform, heNormal
├── layers/
│   ├── Neuron.java
│   ├── Layer.java
│   ├── NeuralNetwork.java        # forward, loss, backprop, fit, save/load
│   ├── ModelFormat.java          # BINARY, JSON, CSV
│   ├── Dropout.java
│   ├── BatchNormalization.java
│   ├── Flatten.java                 # flatten / unflatten, channel-first variants
│   ├── Conv2D.java                  # stride-1 plane SIMD path, backward, parameter copies
│   ├── MaxPool2D.java               # forward with argmax cache, backward
│   ├── Dense.java                   # flat row-major weights, SIMD forward/backward
│   ├── ConvClassifier.java          # Conv-Pool-Conv-Pool-Dense-Dense image classifier
│   └── DenseClassifier.java         # Dense(ReLU) stack, optional Adam on flat parameters
├── losses/
│   ├── LossFunction.java         # MSE, BCE, softmax cross-entropy
│   ├── HingeLoss.java
│   ├── HuberLoss.java
│   ├── KLDivergenceLoss.java
│   ├── CategoricalCrossEntropyLoss.java
│   └── PoissonLoss.java
├── optim/
│   ├── Optimizer.java
│   ├── SGD.java
│   ├── Adam.java
│   ├── RMSProp.java
│   ├── Adagrad.java
│   └── Nadam.java
├── regularizers/
│   ├── L1Regularizer.java
│   ├── L2Regularizer.java
│   └── ElasticNetRegularizer.java
├── training/
│   ├── Trainer.java                 # epochs, batch size, shuffle, seed
│   ├── TrainableClassifier.java     # flat parameter / gradient interface
│   └── DataParallelTrainer.java     # mini-batch SGD split across worker threads
└── vector/
    └── VectorMath.java              # SIMD dot, axpy, axpyRange, dotRange, relu

projects/
├── fashion-mnist/                   # FashionCNNApp (CNN), FashionMnistKerasLike,
│                                    #   FashionMNISTApp (dense MLP, HTML report)
└── mnist/                           # standalone Maven demo
```

## 13.9 Fashion-MNIST projects: arguments and examples

The standalone project `projects/fashion-mnist` has three entry points. Run Maven commands from that folder, after `mvn -B -q install -DskipTests` in the repository root.

| Class | Arguments (positional, all optional) |
| --- | --- |
| `FashionCNNApp` | `trainLimit epochs batchSize testLimit compact threads` (defaults 60000 1 128 10000 true min(4, cores)) |
| `FashionMnistKerasLike` | `trainLimit epochs batchSize testLimit [compact, ignored] threads` (defaults 60000 10 32 10000, threads min(4, cores)) |
| `FashionMNISTApp` | `batchSize` (dense MLP, 10 epochs, saves model and HTML report) |

With Maven, `mvn exec:exec` maps its properties onto the first positional arguments in the order `runArgs`, `cnnEpochs`, `batchSize`, `testLimit`, `compact`. Select the class with `mainClass`. Run `compile` first, because `exec:exec` does not compile:

```
mvn compile exec:exec -DmainClass=FashionCNNApp -DcnnEpochs=5 -DbatchSize=256

mvn compile exec:exec -DmainClass=FashionMnistKerasLike -DrunArgs=60000 -DcnnEpochs=10 -DbatchSize=128 -DtestLimit=10000
```

Running a class directly with `java` also lets you set the thread count:

```
java --add-modules jdk.incubator.vector -cp "target\classes;..\..\target\classes" FashionCNNApp 60000 5 256 10000 true 4
```

> **Vector API everywhere.** The library's convolution and dense layers use `jdk.incubator.vector`, so every run needs `--add-modules jdk.incubator.vector`. The project's `pom.xml` adds it for compile and `exec:exec`.

---

[← 12 · Swing/AWT window](12-swing-ui.md) · [14 · Full code →](14-full-walkthrough.md)
