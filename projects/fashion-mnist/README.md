# Fashion-MNIST project

Two runnable classifiers built on the `neuronflow` library:

- **`FashionCNNApp`**: convolutional network (Conv2D, ReLU, MaxPool2D, two stages, then dense layers). Trains with data-parallel mini-batch SGD and prints per-epoch loss, test accuracy and a per-class report. Weights are kept in memory only; nothing is saved to disk. This is the default.
- **`FashionMNISTApp`**: dense network (784 -> 128 -> 10, Adam). Saves the model to `model/fashion.bin`, `fashion.json`, `fashion.csv` and writes `fashion-eval.html`.

The first run downloads Fashion-MNIST into `data/fashion-mnist`.

## Prerequisites

- JDK 17 and Maven.
- The root library must be installed once, from the repository root:

  ```powershell
  cd C:\Users\dell\Documents\Codex\2026-10-23\nfs
  mvn -B -q install -DskipTests
  ```

- Both the library and this project use `jdk.incubator.vector`. Maven adds the flag automatically. When you start a class with plain `java`, add `--add-modules jdk.incubator.vector` yourself.

## Run with Maven

Run these from `projects\fashion-mnist`:

```powershell
mvn exec:exec                                   # CNN, defaults
mvn exec:exec -DbatchSize=512                   # CNN, batch size 512
mvn exec:exec -DrunArgs=10000 -DcnnEpochs=2     # CNN, 10k images, 2 epochs
mvn exec:exec -Dcompact=false                   # CNN, full 8/16-filter, 128-unit model
mvn exec:exec -DmainClass=FashionMNISTApp -DrunArgs=32   # dense MLP, batch size 32
```

`mvn exec:exec` passes these properties, in this order, to `mainClass`:

| Property | Default | Meaning |
|---|---|---|
| `mainClass` | `FashionCNNApp` | Class to run |
| `runArgs` | `60000` | Training images to use (first N of the training set) |
| `cnnEpochs` | `1` | Number of epochs |
| `batchSize` | `128` | Mini-batch size |
| `testLimit` | `10000` | Number of test images to evaluate (first N) |
| `compact` | `true` | `true`: 4/8 filters, 64 hidden units. `false`: 8/16 filters, 128 hidden units |

The Maven route does not pass a thread count, so `FashionCNNApp` uses `min(4, available processors)` workers. To choose the thread count, run the class directly (below).

`FashionMNISTApp` reads only `runArgs` (as its batch size) and ignores the other properties.

## Run from the command line

First compile the project. The classes read from `target\classes` in this project and from the library in the repository root:

```powershell
cd C:\Users\dell\Documents\Codex\2026-10-23\nfs\projects\fashion-mnist
mvn -B -q compile
```

Then run the class with `java`. The command takes positional arguments:

```
FashionCNNApp [trainLimit] [epochs] [batchSize] [testLimit] [compact] [threads]
```

| Position | Name | Default | Example |
|---|---|---|---|
| 1 | trainLimit | 60000 | `60000` |
| 2 | epochs | 1 | `2` |
| 3 | batchSize | 128 | `512` |
| 4 | testLimit | 10000 | `10000` |
| 5 | compact | true | `true` or `false` |
| 6 | threads | min(4, cores) | `4` |

Every argument is optional, but you must give earlier positions before later ones. For example, to set the thread count you must also give the first five.

PowerShell or cmd.exe (one line):

```powershell
java --add-modules jdk.incubator.vector -cp "target\classes;..\..\target\classes" FashionCNNApp 60000 2 128 10000 true 4
```

Same command split over lines in cmd.exe (`^` continues the line):

```bat
java --add-modules jdk.incubator.vector ^
  -cp "target\classes;..\..\target\classes" ^
  FashionCNNApp 60000 2 128 10000 true 4
```

Examples:

```powershell
# Quick smoke test: 5000 images, one epoch, small test set
java --add-modules jdk.incubator.vector -cp "target\classes;..\..\target\classes" FashionCNNApp 5000 1 128 2000 true

# Full data, 2 epochs, batch 512, full model, 4 threads
java --add-modules jdk.incubator.vector -cp "target\classes;..\..\target\classes" FashionCNNApp 60000 2 512 10000 false 4

# Dense MLP (batch size 32), same class as FashionMNISTApp
java --add-modules jdk.incubator.vector -cp "target\classes;..\..\target\classes" FashionMNISTApp 32

# CNN through the dense app's entry point
java --add-modules jdk.incubator.vector -cp "target\classes;..\..\target\classes" FashionMNISTApp --cnn 60000 2 128 10000 true 4
```

## Output

`FashionCNNApp` prints:

- the parameter count, number of training images, batch size and worker threads;
- one line per epoch with the mean training loss and the epoch time;
- total training time, test accuracy, a 10x10 confusion matrix with precision, recall and F1 per class, and the evaluation time.

Typical results on the laptop used for development (AMD Ryzen 5 3450U, 4 cores):

| Command | Epoch time | Test accuracy |
|---|---|---|
| `FashionCNNApp 60000 2 128 10000 true 4` | about 6 s | about 77% |
| `FashionCNNApp 60000 2 512 10000 true 4` | about 6 s | about 65% |

Batch 512 gives fewer weight updates at the same learning rate (0.01), so it needs more epochs or a larger learning rate to reach the same accuracy.

## Tests

```powershell
mvn -B test
```

This project currently has no unit tests; the library's tests are in the repository root (`mvn -B test` there).

## Files

| Path | Purpose |
|---|---|
| `src/main/java/FashionCNNApp.java` | CNN training and evaluation entry point |
| `src/main/java/FashionMNISTApp.java` | Dense MLP entry point (`--cnn` forwards to the CNN app) |
| `src/main/java/ModelRoundTripTest.java` | Checks that saved models reload and predict identically |
| `pom.xml` | Maven properties and the `exec:exec` configuration |
| `data/fashion-mnist/` | Downloaded dataset (created on first run) |
| `model/` | Saved models from `FashionMNISTApp` |
