# The dataset and its IDX binary format

*Fashion MNIST is a drop-in replacement for the digit MNIST: 70,000 grayscale 28×28 images across 10 clothing classes. Understanding the raw file layout is the key to loading it yourself.*

## 7.1 What is in the dataset

| Split | Images | Files |
| --- | --- | --- |
| Training | 60,000 | `train-images-idx3-ubyte`, `train-labels-idx1-ubyte` |
| Test | 10,000 | `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte` |

### The ten classes

```
0  T-shirt/top    5  Sandal
1  Trouser        6  Shirt
2  Pullover       7  Sneaker
3  Dress          8  Bag
4  Coat           9  Ankle boot
```

Classes 0–4 are upper-body clothing, 5 and 7–9 are footwear (sometimes counted as 3–4–5+7-8-9 "lower body / shoes"), which is why similar items like shirts and pullovers get confused — a useful thing to keep in mind when reading the confusion matrix.

## 7.2 The IDX format

IDX is a tiny header + raw bytes format. Everything is stored **big-endian**, which is exactly Java's `DataInputStream.readInt()` default.

### Image file layout

```
offset  size  meaning
0       4     magic number  = 2051 (0x00000803)
4       4     number of images N
8       4     rows    = 28
12      4     cols    = 28
16      N×784 pixel bytes, row-major, each 0..255

Total size = 16 + N × 784 bytes
For 60,000 images: 16 + 47,040,000 = 47,040,016 bytes
```

### Label file layout

```
offset  size  meaning
0       4     magic number  = 2049 (0x00000801)
4       4     number of labels N
8       N     one unsigned byte per label (0..9)

Total size = 8 + N bytes
For 60,000 labels: 60,008 bytes
```

> **How the loader checks itself.** It compares `Files.size(path)` against the expected length before reading, then verifies end-of-file after reading. Truncated or corrupted downloads fail fast with a clear message instead of silently producing garbage.

## 7.3 Normalization: bytes → [0, 1]

Neural networks train far better when inputs are small and centred-ish. The loader divides each byte by 255:

x = pixelByte / 255.0 → 0 → 0.0, 128 → 0.502, 255 → 1.0

```
double[][] images = new double[count][INPUT_SIZE];

for (int i = 0; i < count; i++) {
    for (int p = 0; p < INPUT_SIZE; p++) {
        images[i][p] = in.readUnsignedByte() / 255.0;
    }
}
```

`readUnsignedByte()` returns 0–255 (not −128..127), and because the dividend `255.0` is a double, the division is floating-point.

## 7.4 One-hot encoding the labels

With mean-squared error the target for class 3 should be a vector that is 1.0 at index 3 and 0.0 elsewhere:

```
/**
 * Converts integer labels to one-hot target vectors.
 *
 * @param labels class indices, each in [0, NUM_CLASSES)
 * @return       [labels.length][NUM_CLASSES] matrix of 0.0/1.0
 */
private static double[][] oneHotEncode(int[] labels) {
    double[][] targets = new double[labels.length][NUM_CLASSES];

    for (int i = 0; i < labels.length; i++) {
        int label = labels[i];

        if (label < 0 || label >= NUM_CLASSES) {
            throw new IllegalArgumentException(
                "Invalid label: " + label);
        }

        targets[i][label] = 1.0;
    }

    return targets;
}
```

label 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

## 7.5 From file to training arrays

**.gz download** → **decompress** → **parse IDX** → **÷255 normalize** → **one-hot labels** → **fit(…)**

---

[← 06 · Training loop](06-training.md) · [08 · Downloader & loader →](08-downloader-loader.md)
