# Exercises to extend the project

*The best way to understand a neural network is to change one thing and watch what happens. Each exercise below is small, self-contained, and touches a single concept from the earlier chapters.*

## 15.1 Warm-up

1. **Print the network shape.** Write a `toString()` on `NeuralNetwork` that prints each layer's size and activation name. Verify it matches `{784, 128, 10}` with `linear, relu, linear`.
2. **Count the parameters.** Add a method `int parameterCount()` that sums `weights.length + 1` over every neuron in every non-input layer. For `{784, 128, 10}` it should return `784·128 + 128 + 128·10 + 10 = 101,770`.
3. **One-hot round-trip.** Given an integer label `c` and `numClasses`, write `float[] oneHot(int c, int numClasses)` and verify `argmax(oneHot(c, n)) == c` for every `c`.

## 15.2 Training behaviour

1. **Learning-rate sweep.** Train with Adam at `lr = 0.0001, 0.001, 0.01, 0.1`. Record final test accuracy and epoch time for each. Which diverges? Which is fastest to reach 80%?
2. **Batch-size effect.** Fix `lr = 0.001` and sweep `batchSize = 1, 16, 32, 128, 1024`. Smaller batches give noisier gradients (sometimes helping escape flat regions); larger batches are more stable but each epoch does fewer updates. Plot accuracy vs. epoch for each.
3. **Shuffle on vs. off.** Train twice with the same seed, once with `shuffle = true` and once with `false`. Compare final accuracy. Why does shuffled data usually win?
4. **Hidden-layer size.** Try topologies `{784, 32, 10}`, `{784, 64, 10}`, `{784, 256, 10}`, `{784, 512, 10}`. Where do returns diminish? Measure both accuracy and epoch time.

## 15.3 Architecture

1. **Add a second hidden layer.** Change the topology to `{784, 128, 64, 10}` and the activations to `{linear, relu, relu, linear}`. Does accuracy improve? (You may need He init on both hidden layers — already handled by `WeightInitializer.heNormal()`.)
2. **Softmax output activation.** Replace the output `linear()` with `Softmax.softmaxActivation()` and switch the loss back to `meanSquaredError()`. Compare with the linear + softmax-cross-entropy baseline. Which converges faster?
3. **Dropout (regularization).** Add a `float dropoutRate` field to `NeuralNetwork`. During training only, zero each hidden activation with probability `dropoutRate` and scale survivors by `1/(1-dropoutRate)`. During prediction, leave activations untouched. Train with `dropoutRate = 0.2` and compare test accuracy — it should improve slightly on the 8,000-example subset.
4. **Learning-rate decay.** Wrap an `Optimizer` that multiplies the learning rate by 0.95 after every epoch. Implement it as a small decorator class implementing `Optimizer`.

## 15.4 Evaluation & reporting

1. **Macro F1.** Compute the unweighted mean of `f1(c)` over all 10 classes (macro-average). Print it alongside overall accuracy.
2. **Top-k accuracy.** Instead of argmax, check whether the true label appears in the top-3 highest-scoring outputs. Print top-1 and top-3 accuracy side by side.
3. **Misclassified gallery.** Extend the HTML report from chapter 11 to list the 20 *worst* mistakes: images where the true class received the lowest score among all 10 outputs.
4. **Per-class image count.** Verify that every row of the confusion matrix sums to exactly 1,000 (the test set has 1,000 images per class). If not, there is a bug in your evaluation loop.

## 15.5 Performance

1. **Benchmark VectorMath.** Write a tiny main that times `VectorMath.dot(a, b, n)` vs. a scalar loop for `n = 784` over 1,000,000 repetitions. Report the speedup.
2. **Batch forward pass.** Add a method `float[][] forwardBatch(float[][] inputs)` that processes an entire mini-batch in one call, reusing the `prevBuffer` across samples. Measure whether it reduces per-epoch time compared to calling `forwardPropagate` in a loop.
3. **Parallel evaluation.** The test-set evaluation loop is embarrassingly parallel. Use a plain `IntStream.range(0, n).parallel()` to compute predicted classes concurrently. Verify the result is identical (it must be — inference is stateless across samples).

## 15.6 Bigger challenges

1. **Train on the full 60,000 examples.** Remove the `TRAIN_LIMIT = 8000` cap and run 20 epochs. Fashion-MNIST test accuracy should reach ~88–89%. Watch the epoch time grow linearly with dataset size.
2. **Save and load weights.** Add `void save(Path)` and `static NeuralNetwork load(Path, ...)` that serialize the flattened parameter vector to a file. Round-trip: save, load, verify identical predictions on 100 test images.
3. **Mini conv layer.** Implement a single 3×3 convolutional layer (stride 1, no padding) that maps `float[784]` → `float[26·26·1]` for one filter, then flatten into the existing MLP. This is the bridge from MLPs to CNNs.
4. **Cross-entropy + softmax gradient check.** Numerically verify the combined softmax+CE gradient: for a random logit vector `z` and one-hot target `t`, compute `(loss(z+εeᵢ) − loss(z−εeᵢ)) / 2ε` for each `i` and compare with `softmaxCrossEntropy().gradient(z, t)[i]`. They should agree to ~5 decimal places.

> **How to verify your work.** Every exercise above changes observable behaviour: accuracy goes up or down, epoch time changes, or a new number appears in the report. If your change produces no difference at all, re-read the relevant chapter — you may have edited a copy instead of the live code path.

---

[← 14 · Full code](14-full-walkthrough.md)
