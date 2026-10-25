# Epochs, mini-batches, and shuffling

*Two classes drive training: `Trainer` holds the hyper-parameters, and `NeuralNetwork.fit` executes the loops. We document each parameter and show how the batches are formed.*

## 6.1 Trainer — the configuration record

```
public class Trainer {
    private final int epochs;
    private final int batchSize;
    private final boolean shuffle;
    private final Random random;

    /**
     * Builds a training configuration.
     *
     * @param epochs     passes over the whole dataset; must be > 0
     * @param batchSize  samples per gradient update; must be > 0
     * @param shuffle    whether to reshuffle row order every epoch
     * @param seed       seed for the shuffling Random, so runs match
     * @throws IllegalArgumentException if epochs or batchSize ≤ 0
     */
    public Trainer(int epochs, int batchSize, boolean shuffle,
                   long seed) {
        if (epochs <= 0 || batchSize <= 0) {
            throw new IllegalArgumentException(
                "Epochs and batch size must be positive");
        }
        this.epochs    = epochs;
        this.batchSize = batchSize;
        this.shuffle   = shuffle;
        this.random    = new Random(seed);
    }

    /** @return number of full passes over the data */
    public int getEpochs()          { return epochs; }

    /** @return samples per gradient update */
    public int getBatchSize()       { return batchSize; }

    /** @return true if rows should be shuffled each epoch */
    public boolean shouldShuffle()  { return shuffle; }

    /** @return the RNG used for shuffling */
    public Random getRandom()       { return random; }
}
```

> **Why a seeded Random?** Without a fixed seed, two runs shuffle differently and produce different results. The seed makes experiments reproducible — a habit worth building early.

## 6.2 fit — the outer loop

```
/**
 * Trains the network on the given dataset.
 *
 * @param trainer epoch/batch/shuffle configuration
 * @param inputs  training examples, [N][inputSize], values in [0,1]
 * @param targets training targets,  [N][outputSize] (usually one-hot)
 * @throws IllegalArgumentException on dimension mismatches
 */
public void fit(Trainer trainer,
                double[][] inputs,
                double[][] targets) {
    validateDataset(inputs, targets);

    int[] order = new int[inputs.length];   // index permutation

    for (int epoch = 0; epoch < trainer.getEpochs(); epoch++) {

        // 1) reset the permutation to identity, then optionally
        //    shuffle it (Fisher–Yates, last element to first)
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }

        if (trainer.shouldShuffle()) {
            Random random = trainer.getRandom();

            for (int i = order.length - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temp  = order[i];
                order[i]  = order[j];
                order[j]  = temp;
            }
        }

        // 2) slice the permuted rows into mini-batches
        int batchSize = trainer.getBatchSize();

        for (int start = 0; start < order.length;
             start += batchSize) {

            int end   = Math.min(start + batchSize, order.length);
            int count = end - start;

            double[][] batchInputs  = new double[count][];
            double[][] batchTargets = new double[count][];

            for (int i = 0; i < count; i++) {
                batchInputs[i]  = inputs[order[start + i]];
                batchTargets[i] = targets[order[start + i]];
            }

            trainBatch(batchInputs, batchTargets);
        }
    }
}
```

### What each loop does

| Concept | Code | Meaning |
| --- | --- | --- |
| **Epoch** | outer `for (epoch …)` | One full pass through all N examples. |
| **Shuffle** | Fisher–Yates on `order` | Randomises the order so mini-batches aren't correlated with data order. |
| **Mini-batch** | inner `for (start …)` | Slices `batchSize` rows; the last slice may be shorter. |
| **Update** | `trainBatch(…)` | Accumulates gradients over the slice, averages, and updates weights once. |

## 6.3 Worked numbers

With Fashion MNIST's 60,000 training images, `batchSize = 128`, and `epochs = 10`:

updates per epoch = ⌈60000 / 128⌉ = 469

 total updates = 469 × 10 = 4690

The final batch has `60000 − 468×128 = 176` samples; `Math.min` handles that tail gracefully.

## 6.4 Typical driver code

```
NeuralNetwork net = new NeuralNetwork(
    new int[]{784, 128, 10},
    new ActivationFunction[]{
        ActivationFunction.linear(),
        ActivationFunction.relu(),
        ActivationFunction.linear()
    },
    LossFunction.meanSquaredError(),
    new Adam(0.001),
    42L
);

Trainer trainer = new Trainer(
    10,      // epochs
    128,     // batch size
    true,    // shuffle every epoch
    42L      // reproducible shuffle seed
);

net.fit(trainer, trainImages, trainTargets);

double loss = net.calculateLoss(trainImages, trainTargets);
System.out.printf("Final loss: %.6f%n", loss);
```

> **calculateLoss** takes `(double[][] inputs, double[][] targets)` and returns the mean loss across all rows — it runs a forward pass per example and averages, without touching any weights. Use it after each epoch to watch progress.

---

[← 05 · Losses & optimizers](05-losses-optimizers.md) · [07 · Fashion MNIST data →](07-data.md)
