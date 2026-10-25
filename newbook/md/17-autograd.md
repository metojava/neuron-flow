# Reverse-mode differentiation on a float tensor

*The `autograd` package is a small PyTorch-style engine: operations on `Tensor` objects record how to propagate gradients backwards, and `backward()` walks that graph in reverse topological order. It is the foundation for the transformer code in the next chapter.*

## 17.1 Tensor

A tensor is a flat row-major `float[]` plus a shape. There are two kinds of tensors:

- **Constants** created by `of`, `zeros`, `randn` and `scalar`; they are not tracked.
- **Parameters** created by `param`, `paramZeros` and `paramRandn`; they set `requiresGrad()` and receive gradients.

```
Tensor x  = Tensor.of(new float[]{1, 2, 3, 4}, 2, 2);   // {2,2}, constant
Tensor w  = Tensor.paramRandn(0.1f, 42L, 2, 3);          // {2,3}, trainable
Tensor b  = Tensor.paramZeros(3);                        // {3},   trainable
Tensor z  = Tensor.zeros(2, 3);                          // constant zeros
Tensor s  = Tensor.scalar(1.0f);                         // {1}
```

| Method | Meaning |
| --- | --- |
| `data()` | Direct read/write access to the flat backing array (used by the optimiser loop). |
| `toFlat()` | Defensive copy of the flat data. |
| `shape()`, `ndim()`, `numel()`, `size(dim)` | Shape metadata; negative dimensions count from the end. |
| `get(i...)`, `set(v, i...)` | Read or write one element with one index per dimension. |
| `grad()` | Gradient buffer, allocated as zeros on first access. |
| `hasGrad()`, `zeroGrad()` | Inspect or clear the gradient buffer. |
| `backward()` | Reverse-mode pass from a scalar tensor. |
| `detach()` | Share data but drop graph tracking. |

## 17.2 How backward works

Each differentiable operation creates an output tensor and attaches two things: its parent tensors and a closure that adds the output gradient into the parents' gradient buffers. `backward()` builds a topological order from the loss, reverses it, seeds the loss gradient with `1.0`, and runs every closure.

forward: y = f(x) → backward: ∂L/∂x += ∂L/∂y · ∂y/∂x

Gradients **accumulate with `+=`**. Call `zeroGrad()` on every parameter between optimiser steps, or deliberately skip it to accumulate across mini-batches. `backward()` requires a scalar; if the tensor has more than one element it throws `IllegalStateException`.

> **No tape reset.** The graph is not freed or detached after `backward()`. Reusing the same loss tensor for a second backward call adds gradients again; rebuild the forward graph for each batch.

## 17.3 Operations

| Category | Methods | Notes |
| --- | --- | --- |
| Arithmetic | `add`, `sub`, `mul`, `scale`, `addScalar` | NumPy broadcasting, aligned from the right. |
| Linear algebra | `matmul` | Batched; last two dimensions are the matrix. |
| Activations | `relu`, `gelu`, `tanh` | GELU uses the tanh approximation. |
| Probability | `softmax` | Numerically stable, over the last dimension. |
| Loss | `crossEntropy` | Fused log-softmax + NLL, returns a scalar mean. |
| Normalisation | `layerNorm` | `gamma` and `beta` are trainable inputs. |
| Embedding | `embedding` | Gathers rows of a `{vocab, D}` weight matrix. |
| Shape | `reshape`, `flattenLeading`, `unflattenLeading`, `transpose` | Row-major data, so reshape never copies logically. |
| Masks | `causalMask` | `{T,T}` with `-1e9` above the diagonal. |
| Indexing | `gatherRows` | Select rows from a 2-D tensor. |

### Broadcasting

Element-wise operations align shapes from the right and treat missing leading dimensions as size 1. For example `{4,3} + {3}` broadcasts the bias across four rows; `{2,1} * {1,5}` produces `{2,5}`. When a parameter is broadcast, its gradient is reduced back to the parameter's original shape.

### Batched matmul

```
{M,K}     x {K,N}     -> {M,N}
{B,M,K}   x {K,N}     -> {B,M,N}     // 2-D right operand broadcasts
{B,M,K}   x {B,K,N}   -> {B,M,N}
{B1,B2,M,K} x {B1,B2,K,N} -> {B1,B2,M,N}
```

### Cross-entropy

`Ops.crossEntropy(logits, targets)` accepts logits of shape `{N,V}` or `{B,T,V}` and flattens all leading dimensions; `targets.length` must equal the number of rows. It subtracts the row maximum before exponentiating, averages the negative log-probability, and its backward pass is the familiar `(softmax - oneHot) / N`.

## 17.4 A complete training loop

There is no `Module` or `Optimizer` wrapper in this package, so a model is just a list of parameter tensors and a forward function. The following two-layer classifier trains on random data:

```
import autograd.Ops;
import autograd.Tensor;

import java.util.List;
import java.util.Random;

int batch = 16;
int features = 4;
int hidden = 8;
int classes = 3;

Tensor w1 = Tensor.paramRandn(0.1f, 1L, features, hidden);
Tensor b1 = Tensor.paramZeros(hidden);
Tensor w2 = Tensor.paramRandn(0.1f, 2L, hidden, classes);
Tensor b2 = Tensor.paramZeros(classes);
List<Tensor> parameters = List.of(w1, b1, w2, b2);

Random random = new Random(7);
float learningRate = 0.05f;

for (int step = 0; step < 500; step++) {
    float[] inputs = new float[batch * features];
    int[] targets = new int[batch];
    for (int i = 0; i < batch; i++) {
        targets[i] = random.nextInt(classes);
        for (int f = 0; f < features; f++) {
            inputs[i * features + f] = random.nextFloat();
        }
    }

    Tensor x = Tensor.of(inputs, batch, features);
    Tensor h = Ops.relu(Ops.add(Ops.matmul(x, w1), b1));
    Tensor logits = Ops.add(Ops.matmul(h, w2), b2);
    Tensor loss = Ops.crossEntropy(logits, targets);

    for (Tensor p : parameters) {
        p.zeroGrad();
    }
    loss.backward();

    for (Tensor p : parameters) {
        float[] values = p.data();
        float[] gradients = p.grad();
        for (int i = 0; i < values.length; i++) {
            values[i] -= learningRate * gradients[i];
        }
    }

    if (step % 100 == 0) {
        System.out.printf("step %d loss %.4f%n", step, loss.get(0));
    }
}
```

## 17.5 What is intentionally missing

- **No parameter registry or `Module`.** Collect parameters yourself or call `parameters()` on the transformer blocks.
- **No optimiser bridge.** `optim.SGD`, `Adam`, `AdamW` and the learning-rate schedules operate on `NeuralNetwork` / flat vectors, not on `autograd.Tensor`.
- **No `no_grad` mode.** Inference still builds a graph unless you wrap work in `detach()` or only use constants.
- **Scalar backward only.** Reduce a vector loss (for example with a mean) before calling `backward()`.
- **float32 only.** There is no double or mixed-precision path.
- **Limited operation set.** Convolution, pooling and recurrent cells live in `layers` and do not participate in this graph.

---

[← 16 · Classical ML](16-classical-ml.md) · [18 · Transformers →](18-transformers.md)
