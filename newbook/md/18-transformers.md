# Encoder and decoder blocks built on autograd

*The `transformers` package assembles attention, layer normalisation and feed-forward layers into encoder (BERT-style) and decoder (GPT-style) stacks. Every parameter is an `autograd.Tensor`, so gradients come from the engine in chapter 17.*

## 18.1 Two transformer implementations

NeuronFlow contains two attention implementations, and they serve different stacks:

| Package | Numeric type | Style | Intended use |
| --- | --- | --- | --- |
| `layers` | `double` | Explicit forward/backward inside `NeuralNetwork` | Attention as a layer in the dense network and CNN demos. |
| `transformers` | `float` | Composable objects returning `Tensor` | Encoder/decoder language models on the autograd engine. |

This chapter covers the second. The classes are `Embedding`, `Linear`, `LayerNorm`, `FeedForward`, `MultiHeadAttention`, `TransformerBlock`, `TransformerEncoder` and `TransformerDecoder`.

## 18.2 Shapes and conventions

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size (number of sequences). |
| `T` | Sequence length (tokens per sequence). |
| `D` | Model width, `dModel`. |
| `H` | Number of attention heads; `D` must be divisible by `H`. |
| `dH` | Head width, `D / H`. |
| `V` | Vocabulary size. |
| `F` | Feed-forward hidden width, usually `4D`. |

Token ids are passed as a flat `int[]` of length `B*T` in row-major order. Hidden states are always `{B,T,D}`. There is no padding mask and no positional embedding inside these classes — see the gaps in section 18.6.

## 18.3 Building blocks

### Embedding

A trainable lookup table of shape `{V,D}`, initialised from `N(0, 0.02²)`. `forward(ids, B, T)` returns `{B,T,D}`; the backward pass scatters gradients back into the rows that were used.

```
Embedding embedding = new Embedding(vocabSize, dModel, seed);
Tensor x = embedding.forward(ids, batchSize, seqLen);
```

### Linear

An affine projection of the last dimension: `out = x @ W + b`. Weights use He-normal initialisation `N(0, 2/inFeatures)`; the bias starts at zero. Leading dimensions are flattened and restored, so the layer works on `{B,T,in}` as well as `{N,in}`.

```
Linear projection = new Linear(dModel, vocabulary, seed);
Tensor logits = projection.forward(hiddenStates);   // {B,T,vocab}
```

### LayerNorm

Normalises the last dimension with learnable `gamma` (initialised to 1) and `beta` (initialised to 0):

y = γ · (x - μ) / √(σ² + ε) + β ε = 1e-5

### FeedForward

Two linear layers with GELU in between: `Linear(D,F) → GELU → Linear(F,D)`. It is applied independently at every position.

```
FeedForward ffn = new FeedForward(dModel, 4 * dModel, seed);
```

### MultiHeadAttention

Self-attention with four projections: query, key, value and output. The input `{B,T,D}` is split into `H` heads of width `dH`, then

Attention(Q,K,V) = softmax(QKT / √dH) V

`forward(x, true)` adds a causal mask, so position `t` can only attend to positions `≤ t`; `forward(x)` and `forward(x, false)` are bidirectional. The mask is a `{T,T}` tensor with `-1e9` above the diagonal, broadcast over batch and heads.

### TransformerBlock

Pre-normalised residual block:

h = x + Attention(LayerNorm(x)) y = h + FeedForward(LayerNorm(h))

The boolean `useCausal` selects decoder-style or encoder-style attention.

### Encoder and decoder stacks

`TransformerEncoder` and `TransformerDecoder` wrap an embedding, a list of `numLayers` blocks and a final layer norm. The encoder passes `false` to every block; the decoder passes `true`. Both expose `parameters()`, which concatenates every trainable tensor in the stack.

```
TransformerEncoder encoder = new TransformerEncoder(
        vocabSize, dModel, numHeads, numLayers, ffnHidden, seed);

TransformerDecoder decoder = new TransformerDecoder(
        vocabSize, dModel, numHeads, numLayers, ffnHidden, seed);

Tensor states = decoder.forward(ids, batchSize, seqLen);   // {B,T,D}
List<Tensor> parameters = decoder.parameters();
```

| Constructor | Parameters |
| --- | --- |
| `Embedding(V, D, seed)` | Vocabulary size, embedding width, RNG seed. |
| `Linear(in, out, seed)` | Input and output widths, RNG seed. |
| `LayerNorm(D)` | Width of the normalised dimension. |
| `FeedForward(D, F, seed)` | Model width and hidden width. |
| `MultiHeadAttention(D, H, seed)` | Model width, head count; requires `D % H == 0`. |
| `TransformerBlock(D, H, F, seed)` | Model width, heads and feed-forward width. |
| `TransformerEncoder(V, D, H, L, F, seed)` | Vocabulary, model width, heads, layers, hidden width. |
| `TransformerDecoder(V, D, H, L, F, seed)` | Same arguments; causal masking enabled. |

## 18.4 Parameter count

For one block, attention has `4·(D² + D)` parameters (four projections with biases), and the feed-forward has `2·(D·F) + D + F`. Add `4·D` for the two layer norms. The stack adds `V·D` for the embedding plus `2·D` for the final norm. For `D=128, H=4, F=512, L=4, V=1000` that is roughly 1.4 million parameters.

## 18.5 A decoder language-model step

The stack returns hidden states, not logits. Add a language-model head and shift the targets so position `t` predicts token `t+1`:

```
import autograd.Ops;
import autograd.Tensor;
import transformers.Linear;
import transformers.TransformerDecoder;

import java.util.List;

int vocab = 256;
int dModel = 32;
int heads = 4;
int layers = 2;
int ffn = 128;
int batch = 4;
int seqLen = 16;

TransformerDecoder decoder = new TransformerDecoder(
        vocab, dModel, heads, layers, ffn, 42L);
Linear lmHead = new Linear(dModel, vocab, 99L);
List<Tensor> parameters = new java.util.ArrayList<>(decoder.parameters());
parameters.addAll(lmHead.parameters());

int[] tokens = new int[batch * seqLen];
// ... fill tokens with ids in [0, vocab) ...
int[] nextTokens = new int[batch * seqLen];
// ... nextTokens[i] = tokens[i + 1] inside each sequence ...

Tensor hidden = decoder.forward(tokens, batch, seqLen);     // {B,T,D}
Tensor logits = lmHead.forward(hidden);                     // {B,T,V}
Tensor loss = Ops.crossEntropy(logits, nextTokens);         // scalar mean

for (Tensor p : parameters) {
    p.zeroGrad();
}
loss.backward();

float learningRate = 0.01f;
for (Tensor p : parameters) {
    float[] values = p.data();
    float[] gradients = p.grad();
    for (int i = 0; i < values.length; i++) {
        values[i] -= learningRate * gradients[i];
    }
}
System.out.println("loss " + loss.get(0));
```

`Ops.crossEntropy` flattens `{B,T,V}` internally, so the target array has length `B*T`. Masking makes the prediction at position `t` depend only on positions `0..t`; that is what makes next-token training valid.

## 18.6 What is missing

- **No positional embeddings.** Both stacks embed tokens only. Without added position information, attention is permutation-invariant. Add a learned `{maxT,D}` table or sinusoidal features before the blocks.
- **No padding mask.** Every sequence in a batch must have the same length, and padding tokens contribute to attention.
- **No dropout.** Regularisation must be added outside the stack.
- **No cross-attention.** The decoder is self-attention only; encoder–decoder models need a second attention sublayer with encoder keys/values.
- **No tokenizer.** `data.text.Tokenizer` is a whitespace vocabulary helper, not a BPE/WordPiece implementation.
- **No training loop or sampler.** There is no fit method, no KV cache, and no temperature/top-k generation.
- **No checkpoint bridge.** `data.model.SafetensorsIO`, `GgufIO` and `OnnxIO` read files, but no loader maps their tensor names onto these classes.

> **Precision.** The transformer stack is float32. For numerical debugging, compare against a double implementation of the same shapes; layer norm's `eps` is fixed at `1e-5`.

---

[← 17 · Autograd](17-autograd.md) · [19 · BERT →](19-bert.md)
