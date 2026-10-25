# ActivationFunction: five functions, one interface

*Without a nonlinear activation, a deep network would collapse into a single linear map. This file defines the interface and five factory methods, each returning a small anonymous class.*

## 4.1 The interface

```
public interface ActivationFunction {

    /**
     * Forward activation: a = f(z).
     *
     * @param x pre-activation value z
     * @return activated value a
     */
    double activate(double x);

    /**
     * Derivative of f expressed in terms of the already-computed
     * output a, i.e. f'(z) given a = f(z). This avoids recomputing.
     *
     * @param output the activated value a cached on the neuron
     * @return f'(z)
     */
    double derivativeFromOutput(double output);

    // ... static factories below
}
```

## 4.2 The five factories

### Sigmoid

```
static ActivationFunction sigmoid() {
    return new ActivationFunction() {
        public double activate(double x) {
            if (x >= 0.0) {
                return 1.0 / (1.0 + Math.exp(-x));
            }
            double e = Math.exp(x);
            return e / (1.0 + e);     // numerically stable branch
        }

        public double derivativeFromOutput(double y) {
            return y * (1.0 - y);
        }
    };
}
```

Maps any real number into (0, 1). The two-branch formula avoids overflow for large negative `x`. Derivative is `σ(z)(1−σ(z))`, which peaks at 0.25 — one reason deep sigmoid nets learn slowly.

### Tanh

```
static ActivationFunction tanh() {
    return new ActivationFunction() {
        public double activate(double x) {
            return Math.tanh(x);
        }

        public double derivativeFromOutput(double y) {
            return 1.0 - y * y;
        }
    };
}
```

Centred at zero, range (−1, 1). Often better than sigmoid in hidden layers because outputs are zero-mean.

### ReLU

```
static ActivationFunction relu() {
    return new ActivationFunction() {
        public double activate(double x) {
            return Math.max(0.0, x);
        }

        public double derivativeFromOutput(double y) {
            return y > 0.0 ? 1.0 : 0.0;
        }
    };
}
```

The default for hidden layers in modern networks: cheap, non-saturating for positive inputs. Note the derivative checks the *output*, which works because `a = 0` exactly when `z ≤ 0`.

### Leaky ReLU

```
static ActivationFunction leakyRelu(double alpha) {
    if (alpha < 0.0) {
        throw new IllegalArgumentException(
            "alpha must be non-negative");
    }

    return new ActivationFunction() {
        public double activate(double x) {
            return x >= 0.0 ? x : alpha * x;
        }

        public double derivativeFromOutput(double y) {
            return y > 0.0 ? 1.0 : alpha;
        }
    };
}
```

Allows a small negative slope `α` so "dead" neurons can still learn. `α = 0.01` is typical.

### Linear

```
static ActivationFunction linear() {
    return new ActivationFunction() {
        public double activate(double x) {
            return x;
        }

        public double derivativeFromOutput(double y) {
            return 1.0;
        }
    };
}
```

Identity function. Used for the input layer (no-op) and as an output activation for regression.

## 4.3 Quick reference

| Function | Range | f′ given output | Best for |
| --- | --- | --- | --- |
| `sigmoid()` | (0, 1) | `y(1−y)` | Binary output |
| `tanh()` | (−1, 1) | `1 − y²` | Small hidden layers |
| `relu()` | [0, ∞) | `y>0 ? 1 : 0` | Hidden layers (default) |
| `leakyRelu(α)` | (−∞, ∞) | `y>0 ? 1 : α` | Avoiding dead units |
| `linear()` | (−∞, ∞) | `1` | Input / regression output |

> **Adding your own.** Implement the two methods anywhere and pass the object into the network constructor — no registration needed. For example a `swish(x) = x·σ(x)` activation is about ten lines.

---

[← 03 · Backpropagation](03-backprop.md) · [05 · Losses & optimizers →](05-losses-optimizers.md)
