# The whole API map, and what is still undocumented

*This chapter is the index to the source tree. It lists every package, says how deeply the book covers it, and records the gaps: classes that exist without a dedicated chapter, and capabilities that are absent rather than merely undocumented. Use it as a checklist when the code moves faster than the prose.*

## 21.1 Three documentation sets

| Location | What it is | Status |
| --- | --- | --- |
| `newbook/` | The current student guide, one HTML chapter per topic plus Markdown in `newbook/md/`. | Current. This is the reference to trust. |
| `docs/` | A formatted archive of three original text files: `first`, `docs` and `explanations`. | Historical. It describes early versions of the library. |
| `README.md` | Build commands, the two main APIs and a short feature tour. | Current summary; not a tutorial. |
| `book/` | An earlier eight-chapter map/reference set. | Superseded by `newbook/`. |

> **Known stale material.** `docs/html/first.html` and `docs/txt/first.txt` show a `Neuron`/`Layer` design with a single learning rate and no activation or optimiser objects. That code no longer matches `layers.NeuralNetwork`. Treat those files as history, not API documentation.

## 21.2 Coverage legend

| Mark | Meaning |
| --- | --- |
| **Full** | A dedicated chapter explains the API, the math and a runnable example. |
| **Partial** | Used in a chapter or the README, but not every public method is described. |
| **Reference** | Listed here with a one-line purpose; no tutorial chapter yet. |
| **Gap** | No class exists for the capability, or the class exists but is not wired into a workflow. |

## 21.3 Package map

| Package | Files | Purpose | Coverage |
| --- | --- | --- | --- |
| `activations` | 12 | Sigmoid, tanh, ReLU family, ELU/SELU, GELU, Swish, Mish, softplus, softmax. | Partial — chapter 04 covers the core five. |
| `autograd` | 2 | `Tensor` and differentiable `Ops`. | Full — chapter 17. |
| `data` | 6 | MNIST/Fashion-MNIST download and IDX loading, float scaling, splitting. | Full — chapters 07–08. |
| `data.augmentation` | 1 | Image augmentation for square float images. | Reference. |
| `data.cv` | 2 | `KFold` and `StratifiedKFold`. | Reference. |
| `data.encoding` | 2 | `LabelEncoder`, `MissingValueImputer` for float arrays. | Reference. |
| `data.io` | 5 | Streaming CSV, JSON Lines, missing-value helpers, Parquet facade. | Reference. |
| `data.io.parquet` | 7 | Pure-JDK flat numeric Parquet reader/writer, Thrift compact metadata, RLE. | Reference. |
| `data.loaders` | 2 | `CsvLoader` and `JsonArrayLoader`. | Reference. |
| `data.model` | 10 | Safetensors, GGUF, ONNX initializers, weight-file abstraction, quantisation. | Reference. |
| `data.pipeline` | 1 | Shuffle, batch and map for `float[][]` datasets. | Reference. |
| `data.scaling` | 2 | Min-max and robust scaling for `float[][]`. | Reference. |
| `data.text` | 1 | Whitespace tokenizer with a vocabulary. | Reference. |
| `evaluation` | 5 | Accuracy/confusion/P/R/F1, regression metrics, AUC, HTML report, PNG encoding. | Partial — chapters 10–11 cover classification and reports. |
| `evaluation.bootstrap` | 1 | Bootstrap confidence intervals for a metric. | Reference. |
| `evaluation.nlp` | 2 | BLEU and ROUGE. | Reference. |
| `examples` | 4 | Fashion-MNIST, MNIST, classical-ML demo, dispatcher. | Partial — chapters 13–16. |
| `hyperparam` | 1 | Grid and random search spaces. | Reference. |
| `initialization` | 1 | Xavier uniform and He normal initialisers. | Partial — used in chapters 01 and 09. |
| `layers` | 37 | Dense network, convolutions, pooling, normalisation, dropout, recurrent cells, attention, sequential API, model I/O. | Partial — chapters 01–03 and 09 cover the dense core; CNN/RNN/attention layers are reference-only. |
| `logging` | 1 | CSV training logger. | Reference. |
| `losses` | 9 | MSE, BCE, cross-entropy, hinge, Huber, KL, MAE, Poisson, label smoothing. | Partial — chapter 05 covers the core losses. |
| `ml` | 9 | Frames, vectors, pipelines, estimators, clustering, splits. | Full — chapter 16. |
| `ml.kernel` | 3 | Linear and RBF kernels for the SVM. | Full — chapter 16. |
| `ml.metrics` | 1 | Accuracy, MSE and R² for the classical API. | Full — chapter 16. |
| `ml.model` | 7 | Linear/logistic regression, decision tree, random forest, SVM, k-means, shared validation. | Full — chapter 16. |
| `ml.preprocessing` | 5 | Imputer, one-hot encoder, standard and min-max scalers. | Full — chapter 16. |
| `optim` | 12 | SGD, momentum, NAG, Adam, AdamW, Nadam, RMSProp, Adagrad, clipping, noise, schedules. | Partial — chapter 05 covers SGD and Adam. |
| `regularizers` | 3 | L1, L2 and elastic-net penalties. | Reference. |
| `training` | 4 | Trainer configuration, gradient accumulation, data-parallel training, trainable-classifier abstraction. | Partial — chapter 06 covers `Trainer`. |
| `training.callbacks` | 4 | Early stopping, checkpointing, plateau scheduler, callback interface. | Gap — not invoked by `Trainer`. |
| `transformers` | 8 | Embedding, linear, layer norm, feed-forward, attention, blocks, encoder and decoder stacks. | Full — chapter 18; BERT and GPT gaps in 19–20. |
| `util` | 3 | Class weights, progress bar, seed manager. | Reference. |
| `vector` | 1 | Vector API dot, ReLU and scaled-add helpers. | Partial — chapter on performance. |

## 21.4 Classes with no dedicated tutorial yet

### Activations

Chapter 04 documents sigmoid, tanh, ReLU, Leaky ReLU and linear. The package also contains `ELU`, `SELU`, `GELU`, `Swish`, `HardSwish`, `Mish`, `PReLU`, `ReLU6`, `Softplus` and `Softmax`, plus the `Activations` registry. Their derivatives exist; use them through `ActivationFunction` rather than calling the classes directly unless you need a parameterised activation such as `PReLU`.

### Layers beyond the dense core

The following are implemented and tested but not taught step by step:

- **Convolution and pooling:** `Conv2D`, `DilatedConv2D`, `SeparableConv2D`, `GroupedConv2D`, `TransposedConv2D`, `MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D`.
- **Normalisation:** `BatchNormalization`, `BatchNormLayer`, `LayerNorm`, `GroupNorm`, `InstanceNorm`.
- **Sequence models:** `RNN`, `LSTM`, `GRU`, `RecurrentCell`, `RecurrentStack`.
- **Attention and residuals:** `SelfAttention`, `MultiHeadAttention`, `ResidualBlock`, `Embedding`.
- **High-level models:** `DenseClassifier`, `ConvClassifier`, `Sequential`, `BatchPredictor`.
- **Persistence:** `ModelFormat`, `ModelLoader`, `ModelSummary`, `WeightImporter`.

> **Scope note.** These classes are `double`-based and live inside the `NeuralNetwork` forward/backward contract. `WeightImporter` maps safetensors/GGUF/ONNX tensors into `layers.Dense`, `Conv2D`, `Embedding`, `LayerNorm`, recurrent cells and the two attention layers. It does not target `transformers.*`.

### Losses, optimisers, regularisers

`losses` adds `HingeLoss`, `HuberLoss`, `KLDivergenceLoss`, `CategoricalCrossEntropyLoss`, `MAELoss`, `PoissonLoss` and `LabelSmoothing` to the core MSE/BCE/cross-entropy set. `optim` adds `NAG`, `AdamW`, `Nadam`, `RMSProp`, `Adagrad`, `GradientClipper`, `GradientNoise`, `LearningRateSchedule` and the `Optimizers` registry. `regularizers` provides L1, L2 and elastic net. Only SGD and Adam are walked through in chapter 05; the others follow the same `Optimizer` interface.

### Data and model I/O

These are working utilities rather than demos:

- `CsvStreamReader`/`CsvStreamWriter`, `JsonLinesStream`, `CsvLoader`, `JsonArrayLoader`.
- `ParquetIO` and the pure-JDK `data.io.parquet` reader/writer. The supported profile is flat numeric columns; it is not a general Parquet implementation.
- `SafetensorsIO`, `GgufIO`, `OnnxIO`, `OnnxGraph`, `WeightFile` and `GgufQuant`. ONNX support reads graph metadata and initializers; it does not execute ONNX graphs.
- `data.text.Tokenizer` is a whitespace tokenizer, not BPE or WordPiece.

### Evaluation, tuning and utilities

`ClassificationMetrics` (confusion matrix, per-class precision/recall/F1), `RegressionMetrics`, `AucMetrics`, `Bootstrap`, `BleuScore` and `RougeScore` are available. `GridSearch` builds parameter combinations and supports random search, but does not run training for you. `ClassWeights`, `ProgressBar`, `SeedManager` and `TrainingLogger` are small helpers.

## 21.5 Capabilities that are absent, not just undocumented

| Capability | Status | Closest existing piece |
| --- | --- | --- |
| BERT model, WordPiece, MLM/NSP heads | Absent | `TransformerEncoder`, chapter 19 |
| GPT model, BPE, sampling, KV cache | Absent | `TransformerDecoder`, chapter 20 |
| Positional and segment embeddings | Absent | Add a second `Embedding` (chapter 19) |
| Autograd optimiser bridge | Absent | Manual SGD loop (chapter 17); `optim.*` targets `NeuralNetwork` |
| Dropout / masking inside transformers | Absent | No dropout or padding mask in `transformers.*` |
| Cross-attention encoder–decoder | Absent | `TransformerBlock` is self-attention only |
| Callback-driven training | Not wired | Callback classes exist; `Trainer` ignores them |
| ONNX graph execution | Absent | `OnnxIO` reads initializers and graph metadata only |
| Pretrained transformer weights | Not mapped | `SafetensorsIO`/`GgufIO` can read files; no `transformers.*` mapping |
| GPU or mixed precision | Absent | CPU float32 only |
| General-purpose Parquet (strings, nesting, all codecs) | Limited | `ParquetIO` supports flat numeric columns |
| Classical-ML model persistence | Absent | `ml` models have no save/load methods |
| Feature importance for linear models | Absent | Coefficients are exposed; no ranking helper |
| Probability calibration | Absent | Sigmoid outputs from logistic/SVM are uncalibrated |

## 21.6 Documentation debt, ranked

1. **Update the archived introduction.** `docs/html/*.html` and `docs/txt/*.txt` still describe the original four-class design. A banner pointing to `newbook/` now exists, but the listings themselves are historical.
2. **Write a CNN chapter.** `Conv2D`, pooling, `ConvClassifier` and `WeightImporter` are substantial and currently only reference-level.
3. **Write a recurrent chapter.** `RNN`, `LSTM`, `GRU` and `RecurrentStack` have tests but no walkthrough.
4. **Document the optimiser family.** One page comparing SGD, momentum, NAG, Adam, AdamW, Nadam, RMSProp and Adagrad, including when each helps.
5. **Document data I/O.** CSV/JSONL/Parquet/Safetensors/GGUF/ONNX with small round-trip examples.
6. **Wire the callbacks.** Either integrate `training.callbacks` into `Trainer` or state clearly that they are standalone building blocks.
7. **Document evaluation beyond accuracy.** AUC, bootstrap intervals, BLEU/ROUGE and the regression metrics.
8. **Add a migration page.** Map the names in `docs/` (`Neuron`, `Layer`, single learning rate) to the current API.

## 21.7 Build and run reference

```
# compile and test (enables the incubating Vector API)
mvn clean test

# run all tests quietly
mvn -q test

# Fashion-MNIST demo (default)
mvn exec:exec

# dispatcher: fashion | mnist | ml
java --add-modules jdk.incubator.vector -cp target/classes examples.Main ml

# classical-ML demo directly
java --add-modules jdk.incubator.vector -cp target/classes examples.ClassicalMlDemo
```

Tests are under `src/test/java`; the classical-ML suites are `ml.DataFrameTest`, `ml.PreprocessingTest`, `ml.SupervisedModelsTest` and `ml.KMeansTest`. The `autograd` and `transformers` packages compile but currently have no dedicated unit tests; they are documented from the source contracts in chapters 17–20.

> **Keeping this page honest.** When a new package or public class lands, add it to the package map and either write a chapter or mark it Reference. The map is deliberately one table so drift is easy to spot.

---

[← 20 · GPT](20-gpt.md) · [Start here →](index.md)
