# Twelve standalone tutorials

*Each project is a self-contained tutorial that builds one complete application on top of the library. Start at 01 and work up, or jump to whichever problem interests you — they only depend on the core classes (`NeuralNetwork`, `Trainer`, `LossFunction`, `Adam`), not on each other.*

## The projects

- **[01 · XOR](p01.md)** — Four examples, one hidden layer. The hello-world of neural networks.
- **[02 · MNIST digits](p02.md)** — Handwritten digit recognition on the original MNIST dataset.
- **[03 · Fashion-MNIST](p03.md)** — The main demo: train, evaluate, confusion matrix, HTML report.
- **[04 · Breast cancer](p04.md)** — Binary classification on the Wisconsin dataset (30 features).
- **[05 · Housing prices](p05.md)** — Regression: predict Boston-area house prices from 13 features.
- **[06 · IMDB sentiment](p06.md)** — Binary text classification with bag-of-words features.
- **[07 · Autoencoder](p07.md)** — Learn a compressed representation and reconstruct the input.
- **[08 · Character-level text](p08.md)** — Generate text one character at a time from a training corpus.
- **[09 · Stock prediction](p09.md)** — Time-series regression with a sliding window.
- **[10 · Conv classifier](p10.md)** — Conv2D + MaxPool2D layers for image classification.
- **[11 · Multi-armed bandit](p11.md)** — Reinforcement learning with epsilon-greedy action selection.
- **[12 · GAN](p12.md)** — Generator and discriminator trained adversarially.

## Prerequisites

- Read chapters 01–09 of the book first — they explain every class these projects use.
- Projects 02, 03, 10 download their dataset automatically on first run.
- Projects 04–09 use small built-in datasets or CSV files you provide.

## How to run a project

```
# install the library once
mvn install -DskipTests

# then run any project
cd projects/fashion-mnist
mvn compile
mvn exec:exec -DbatchSize=64
```

Every standalone project has its own `pom.xml` with the `--add-modules jdk.incubator.vector` flag pre-configured, so `mvn exec:exec` just works.

---

[← Start here](../index.md) · [01 · XOR →](p01.md)
