# Measuring error and updating weights

*A loss function turns "how wrong" into a number and its gradient. An optimizer turns that gradient into better parameters. Both are small interfaces with two or three methods.*

## 5.1 LossFunction

```
public interface LossFunction {

    /**
     * Scalar error for one prediction/target pair.
     *
     * @param predictions network outputs, length N
     * @param targets     desired outputs,  length N
     * @return loss value (smaller is better)
     */
    double loss(double[] predictions, double[] targets);

    /**
     * Gradient of the loss with respect to the predictions.
     *
     * @param predictions network outputs, length N
     * @param targets     desired outputs,  length N
     * @return one partial derivative per output unit
     */
    double[] gradient(double[] predictions, double[] targets);

    // static factories: meanSquaredError(), binaryCrossEntropy()
}
```

### Mean squared error

```
static LossFunction meanSquaredError() {
    return new LossFunction() {
        public double loss(double[] p, double[] t) {
            checkShapes(p, t);
            double sum = 0.0;
            for (int i = 0; i < p.length; i++) {
                double d = p[i] - t[i];
                sum += d * d;
            }
            return sum / p.length;
        }

        public double[] gradient(double[] p, double[] t) {
            checkShapes(p, t);
            double[] g = new double[p.length];
            for (int i = 0; i < p.length; i++) {
                g[i] = 2.0 * (p[i] - t[i]) / p.length;
            }
            return g;
        }
    };
}
```

L = (1/N) Σ (pᵢ − tᵢ)² → ∂L/∂pᵢ = 2(pᵢ − tᵢ)/N

This is what `FashionMNISTRunner` uses with one-hot targets. The division by `N` keeps gradients on a comfortable scale.

### Binary cross-entropy

```
static LossFunction binaryCrossEntropy() {
    return new LossFunction() {
        public double loss(double[] p, double[] t) {
            checkShapes(p, t);
            double sum = 0.0;
            double eps = 1e-12;   // guards log(0)

            for (int i = 0; i < p.length; i++) {
                if (t[i] < 0.0 || t[i] > 1.0) {
                    throw new IllegalArgumentException(
                        "Binary targets must be in [0, 1]");
                }
                double q = Math.max(eps, Math.min(1.0 - eps, p[i]));
                sum -= t[i] * Math.log(q)
                     + (1.0 - t[i]) * Math.log(1.0 - q);
            }
            return sum / p.length;
        }

        public double[] gradient(double[] p, double[] t) {
            checkShapes(p, t);
            double[] g = new double[p.length];
            double eps = 1e-12;
            for (int i = 0; i < p.length; i++) {
                double q = Math.max(eps, Math.min(1.0 - eps, p[i]));
                g[i] = (-t[i] / q + (1.0 - t[i]) / (1.0 - q))
                       / p.length;
            }
            return g;
        }
    };
}
```

Pairs with a sigmoid output and targets in [0, 1]. The epsilon clamp prevents `log(0) = −∞`.

## 5.2 Optimizer

```
public interface Optimizer {
    /**
     * Updates parameters in place from their gradients.
     *
     * @param parameters flat weight+bias array (modified by the call)
     * @param gradients  matching array of ∂L/∂parameter
     */
    void update(double[] parameters, double[] gradients);
}
```

### SGD with optional momentum

```
public class SGD implements Optimizer {
    private final double learningRate;
    private final double momentum;   // 0.0 = plain SGD
    private double[] velocity;       // state, grows lazily

    /** Plain SGD. @param learningRate step size, must be > 0 */
    public SGD(double learningRate) {
        this(learningRate, 0.0);
    }

    /**
     * SGD with momentum.
     *
     * @param learningRate step size, must be > 0
     * @param momentum     [0, 1); 0.9 is a common value
     */
    public SGD(double learningRate, double momentum) {
        if (!(learningRate > 0.0) ||
            momentum < 0.0 || momentum >= 1.0) {
            throw new IllegalArgumentException("Invalid SGD parameters");
        }
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    /**
     * v ← momentum·v + g ;  θ ← θ − learningRate·v
     *
     * @param parameters weights (updated in place)
     * @param gradients  their gradients
     */
    @Override
    public void update(double[] parameters, double[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException(
                "Parameter/gradient mismatch");
        }
        if (velocity == null || velocity.length != parameters.length) {
            velocity = new double[parameters.length];
        }

        for (int i = 0; i < parameters.length; i++) {
            velocity[i] = momentum * velocity[i] + gradients[i];
            parameters[i] -= learningRate * velocity[i];
        }
    }
}
```

### Adam

```
public class Adam implements Optimizer {
    private final double learningRate;
    private final double beta1;    // default 0.9
    private final double beta2;    // default 0.999
    private final double epsilon;  // default 1e-8

    private double[] firstMoment;  // m: mean of gradients
    private double[] secondMoment; // v: mean of squared gradients
    private long timestep;         // for bias correction

    /** Adam with the standard hyper-parameters. */
    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    /**
     * Full Adam configuration.
     *
     * @param learningRate step size, must be > 0
     * @param beta1        first-moment decay in [0, 1)
     * @param beta2        second-moment decay in [0, 1)
     * @param epsilon      small constant avoiding division by zero
     */
    public Adam(double learningRate, double beta1,
                double beta2, double epsilon) {
        // ... validation omitted for brevity
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    /**
     * One Adam step:
     *   m ← β₁m + (1−β₁)g
     *   v ← β₂v + (1−β₂)g²
     *   m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ)
     *   θ ← θ − lr·m̂/(√v̂ + ε)
     *
     * @param parameters weights (updated in place)
     * @param gradients  their gradients
     */
    @Override
    public void update(double[] parameters, double[] gradients) {
        // ... validation omitted for brevity

        timestep++;
        double correction1 = 1.0 - Math.pow(beta1, timestep);
        double correction2 = 1.0 - Math.pow(beta2, timestep);

        for (int i = 0; i < parameters.length; i++) {
            double g = gradients[i];

            firstMoment[i]  = beta1 * firstMoment[i]
                            + (1.0 - beta1) * g;
            secondMoment[i] = beta2 * secondMoment[i]
                            + (1.0 - beta2) * g * g;

            double mHat = firstMoment[i]  / correction1;
            double vHat = secondMoment[i] / correction2;

            parameters[i] -= learningRate * mHat
                           / (Math.sqrt(vHat) + epsilon);
        }
    }
}
```

> **State matters.** Adam keeps `firstMoment`, `secondMoment` and `timestep` across calls. Create *one* optimizer per network — sharing an instance between two networks would mix their statistics. The first call also lazily sizes the arrays, so a parameter-count change is handled gracefully.

### Choosing

| Optimizer | Needs | Notes |
| --- | --- | --- |
| `SGD(0.1)` | well-scaled data | Simple; add momentum for faster convergence. |
| `SGD(0.1, 0.9)` | noise tolerance | Momentum smooths jitter. |
| `Adam(0.001)` | almost everything | Default in this project; insensitive to gradient scale. |

---

[← 04 · Activations](04-activations.md) · [06 · Training loop →](06-training.md)
