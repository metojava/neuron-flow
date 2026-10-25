# Accuracy, per-class errors, and reading the matrix

*After training you need to know *how well* the model does and *where* it fails. Three small methods do this: one returns a number, one prints a matrix, one picks the class.*

## 10.1 argmax — outputs → class id

```
/**
 * Returns the index of the largest output value.
 *
 * @param values network outputs, e.g. 10 linear scores
 * @return index of the maximum, i.e. the predicted class
 * @throws IllegalArgumentException if the array is null or empty
 */
private static int argmax(double[] values) {
    if (values == null || values.length == 0) {
        throw new IllegalArgumentException("Output must not be empty");
    }

    int best = 0;

    for (int i = 1; i < values.length; i++) {
        if (values[i] > values[best]) {
            best = i;
        }
    }

    return best;
}
```

Ties resolve to the lowest index because the comparison is strictly greater-than. With linear outputs, argmax is equivalent to taking the softmax first — softmax is monotonic, so it never changes which index is largest.

## 10.2 evaluateAccuracy — a single number

```
/**
 * Classification accuracy over a labelled set.
 *
 * @param network trained network
 * @param images  examples, [N][784]
 * @param labels  true class ids, [N]
 * @return fraction correct, in [0, 1]
 * @throws IllegalArgumentException if the arrays are empty or
 *                                  their lengths differ
 */
private static double evaluateAccuracy(NeuralNetwork network,
                                     double[][] images,
                                     int[] labels) {
    if (images.length != labels.length || images.length == 0) {
        throw new IllegalArgumentException("Invalid evaluation dataset");
    }

    int correct = 0;

    for (int i = 0; i < images.length; i++) {
        int predicted = argmax(network.predict(images[i]));

        if (predicted == labels[i]) {
            correct++;
        }
    }

    return (double) correct / (double) images.length;
}
```

accuracy = correct / N → 8407 / 10000 = 0.8407 = 84.07%

## 10.3 evaluate — accuracy plus confusion matrix

```
/**
 * Prints overall accuracy and the full confusion matrix.
 *
 * Matrix convention: rows = actual class, columns = predicted class,
 * so confusion[actual][predicted] counts every example.
 *
 * @param network trained network
 * @param images  test examples, [N][784]
 * @param labels  test labels, [N]
 */
private static void evaluate(NeuralNetwork network,
                           double[][] images,
                           int[] labels) {

    int[][] confusion = new int[CLASSES][CLASSES];
    int correct = 0;

    for (int i = 0; i < images.length; i++) {
        int predicted = argmax(network.predict(images[i]));
        int actual    = labels[i];

        confusion[actual][predicted]++;

        if (predicted == actual) {
            correct++;
        }
    }

    System.out.printf("Test accuracy: %.2f%% (%d/%d)%n",
        100.0 * correct / images.length, correct, images.length);

    System.out.println(
        "\nConfusion matrix (rows = actual, cols = predicted):");

    // header: predicted class numbers
    System.out.print("           ");
    for (int c = 0; c < CLASSES; c++) {
        System.out.printf("%6d", c);
    }
    System.out.println();

    // one row per actual class, labelled with its class name
    for (int actual = 0; actual < CLASSES; actual++) {
        System.out.printf("%-12s ", CLASS_NAMES[actual]);

        for (int predicted = 0; predicted < CLASSES; predicted++) {
            System.out.printf("%6d", confusion[actual][predicted]);
        }

        System.out.println();
    }
}
```

## 10.4 Example output

```
Test accuracy: 84.07% (8407/10000)

Confusion matrix (rows = actual, cols = predicted):
                0     1     2     3     4     5     6     7     8     9
T-shirt/top    779     5    31    47     2     3   119     0    14     0
Trouser          2   960     4    27     3     0     3     0     1     0
Pullover         8     4   767    10   115     0    84     0    12     0
Dress           33    12    17   851    36     1    45     0     5     0
Coat            1     4   142    34   704     0   109     0     6     0
Sandal          1     0     0     2     0   899     1    57     3    37
Shirt          120     2   129    35    87     0   606     0    21     0
Sneaker         0     0     0     0     0    24     0   935     0    41
Bag             1     1     6     6     1     2    19     4   958     2
Ankle boot      0     0     0     0     0     7     1    44     0   948
```

### How to read it

- **Diagonal** = correct predictions. Sum of diagonal = 8,407 → matches the reported accuracy.
- **Off-diagonal in a row** = that actual class mistaken for something else. Row 0 (T-shirt/top): 119 became "Shirt" — expected, they look alike.
- **Off-diagonal in a column** = which true classes get pulled into a prediction. Column 6 (Shirt) collects items from many rows.
- **Rows sum to the class count** (1,000 per class in this test set), so you can eyeball per-class recall: `779/1000 = 77.9%` for T-shirt/top, `960/1000 = 96.0%` for Trouser.

> **Per-class recall formula.** For class c: `recall(c) = confusion[c][c] / Σₖ confusion[c][k]`. Precision instead divides by a column: `precision(c) = confusion[c][c] / Σₖ confusion[k][c]`. Both are one line of code over the matrix — a good exercise in chapter 15.

## 10.5 Printing class names instead of numbers

The matrix header uses indices for compactness, while the row labels use names from `FashionMNISTLoader.CLASS_NAMES`. For sample predictions the runner prints both:

```
System.out.printf("Predicted: %-15s (%d) | Actual: %-15s (%d)%n",
    CLASS_NAMES[predicted], predicted,
    CLASS_NAMES[actual],    actual);

// Predicted: Shirt           (6) | Actual: Dress           (3)
```

---

[← 09 · Keras-like classifier](09-keras-like.md) · [11 · HTML report →](11-html-report.md)
