# Understand every line of a Java neural network

*This book explains the actual code in this folder — `Neuron`, `Layer`, `NeuralNetwork`, `Trainer`, `ActivationFunction`, `LossFunction`, `SGD`, `Adam`, and the Fashion-MNIST downloader/loader/runner — with method-by-method documentation, parameter tables, the math behind each step, and complete code examples you can run. The later chapters add the two newer APIs: a scikit-learn-style `ml` package for tabular data (chapter 16) and an autograd/transformer stack for sequence models (chapters 17–20). Chapter 21 maps every package and lists what is still missing.*

## What you will learn

- **[01 · Foundations](01-foundations.md)** — Neuron, Layer, the neural-network constructor.
- **[02 · Forward pass](02-forward-pass.md)** — How an input becomes a prediction.
- **[03 · Backpropagation](03-backprop.md)** — Gradients, chain rule, parameter updates.
- **[04 · Activations](04-activations.md)** — Sigmoid, tanh, ReLU, Leaky ReLU, linear.
- **[05 · Losses & optimizers](05-losses-optimizers.md)** — MSE, BCE, SGD, Adam.
- **[06 · Training loop](06-training.md)** — Epochs, mini-batches, shuffling.
- **[07 · Fashion MNIST data](07-data.md)** — IDX format, normalization, one-hot.
- **[08 · Downloader & loader](08-downloader-loader.md)** — Download from GitHub, parse IDX.
- **[09 · Keras-like classifier](09-keras-like.md)** — Flatten → Dense(128, ReLU) → Dense(10).
- **[10 · Evaluation](10-evaluation.md)** — Accuracy, confusion matrix, per-class.
- **[11 · HTML report](11-html-report.md)** — Save results as a web page.
- **[12 · Swing/AWT window](12-swing-ui.md)** — Show images and predictions live.
- **[13 · Run it](13-run.md)** — Compile and run with Maven/CLI.
- **[14 · Full code](14-full-walkthrough.md)** — Complete listing of all classes.
- **[15 · Practice](15-practice.md)** — Exercises to extend the project.
- **[16 · Classical machine learning](16-classical-ml.md)** — DataFrame, pipelines, regression, trees, SVM, k-means.
- **[17 · Autograd](17-autograd.md)** — Tensor, differentiable ops, backward pass.
- **[18 · Transformers](18-transformers.md)** — Attention, encoder and decoder stacks.
- **[19 · BERT](19-bert.md)** — Encoder-only models: what exists, what is missing.
- **[20 · GPT](20-gpt.md)** — Decoder-only language modelling and generation.
- **[21 · API reference](21-api-reference.md)** — Every package, coverage status, and documentation gaps.
- **[Performance](performance.md)** — float conversion, Vector API, SIMD speedups.
- **[Projects](projects/index.md)** — 12 standalone tutorials: XOR to GAN.

## The plan

**Read** (the code) → **Understand** (the math) → **Download** (Fashion MNIST) → **Train** (the classifier) → **Evaluate** (& report)

> **Prerequisites.** JDK 17, a text editor, and basic Java (arrays, loops, classes). You do not need TensorFlow or any ML library — everything is plain Java. Maven is optional but convenient for running the examples.

## Project layout

```
src/main/java/
├── activations/           # sigmoid, tanh, ReLU family, ELU/SELU, GELU,
│                          # Swish, Mish, PReLU, Softmax, registry
├── autograd/              # Tensor, Ops: reverse-mode automatic differentiation
├── data/
│   ├── MNISTDownloader / MNISTLoader
│   ├── FashionMNISTDownloader / FashionMNISTLoader
│   ├── DatasetSplit, StandardScaler
│   ├── augmentation/      # image augmentation
│   ├── cv/                # KFold, StratifiedKFold
│   ├── encoding/          # LabelEncoder, MissingValueImputer
│   ├── pipeline/          # shuffle, batch, map for float[][]
│   ├── scaling/           # MinMaxScaler, RobustScaler
│   ├── text/              # whitespace Tokenizer
│   ├── io/                # CSV, JSONL, Parquet
│   └── model/             # Safetensors, GGUF, ONNX, WeightFile, quantisation
├── evaluation/            # classification/regression metrics, AUC, HTML report
│   ├── bootstrap/         # confidence intervals
│   └── nlp/               # BLEU, ROUGE
├── examples/              # FashionMNISTRunner, MNISTRunner, ClassicalMlDemo, Main
├── hyperparam/            # GridSearch (grid and random search spaces)
├── initialization/        # WeightInitializer (Xavier, He)
├── layers/                # dense core + CNN, pooling, norm, dropout,
│                          # RNN/LSTM/GRU, attention, Sequential, model I/O
├── logging/               # TrainingLogger
├── losses/                # MSE, BCE, cross-entropy, hinge, Huber, KL, MAE,
│                          # Poisson, label smoothing
├── ml/
│   ├── DataFrame, Vector, Pipeline, DatasetSplit, Estimator/Transformer
│   ├── preprocessing/     # imputer, one-hot encoder, standard/min-max scalers
│   ├── model/             # linear/logistic regression, tree, forest, SVM, k-means
│   ├── kernel/            # linear and RBF kernels
│   └── metrics/           # accuracy, MSE, R²
├── optim/                 # SGD, momentum, NAG, Adam, AdamW, Nadam, RMSProp,
│                          # Adagrad, clipping, noise, schedules
├── regularizers/          # L1, L2, elastic net
├── training/              # Trainer, gradient accumulation, data parallel
│   └── callbacks/         # early stopping, checkpoint, plateau scheduler
├── transformers/          # Embedding, Linear, LayerNorm, attention,
│                          # TransformerBlock/Encoder/Decoder
├── util/                  # class weights, progress bar, seed manager
└── vector/                # SIMD helpers using the Vector API

projects/
├── fashion-mnist/         # standalone Maven demo
└── mnist/                 # standalone Maven demo
```

---

[01 · Foundations →](01-foundations.md)
