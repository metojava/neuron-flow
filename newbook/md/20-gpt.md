# Decoder-only language models and greedy generation

*There is no `Gpt` class in NeuronFlow, but the `transformers.TransformerDecoder` is deliberately GPT-style: token embeddings, stacked pre-norm blocks with causal self-attention, a final layer norm, and a separate language-model head. This chapter covers training a next-token predictor and generating text from it.*

## 20.1 The decoder is the GPT backbone

Each `TransformerBlock` inside the decoder receives `useCausal=true`. Query position `t` therefore sees keys and values only from positions `0..t`, which is exactly the autoregressive constraint. The forward pass is:

x = Embedding(ids) → [Block1 … BlockL] → LayerNorm → Linear(V)

Everything after the final layer norm is task-specific. For language modelling, project `{B,T,D}` to `{B,T,V}` and train position `t` to predict token `t+1`.

```
TransformerDecoder decoder = new TransformerDecoder(
        vocabSize, dModel, numHeads, numLayers, ffnHidden, seed);
Linear lmHead = new Linear(dModel, vocabSize, seed + 1);

Tensor hidden = decoder.forward(tokens, batch, seqLen);   // {B,T,D}
Tensor logits = lmHead.forward(hidden);                   // {B,T,V}
Tensor loss = Ops.crossEntropy(logits, nextTokens);       // scalar
```

`nextTokens[i]` is the token that follows `tokens[i]` inside the same sequence; do not cross a sequence boundary when building the target array.

## 20.2 A complete training step

```
import autograd.Ops;
import autograd.Tensor;
import transformers.Linear;
import transformers.TransformerDecoder;

import java.util.ArrayList;
import java.util.List;

int vocab = 512;
int dModel = 64;
int heads = 4;
int layers = 3;
int ffn = 256;

TransformerDecoder decoder = new TransformerDecoder(
        vocab, dModel, heads, layers, ffn, 42L);
Linear head = new Linear(dModel, vocab, 7L);

List<Tensor> parameters = new ArrayList<>(decoder.parameters());
parameters.addAll(head.parameters());

float learningRate = 0.003f;

// one mini-batch: int[] tokens, int[] nextTokens, both length B*T
Tensor x = decoder.forward(tokens, batch, seqLen);
Tensor logits = head.forward(x);
Tensor loss = Ops.crossEntropy(logits, nextTokens);

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

System.out.printf("loss %.4f%n", loss.get(0));
```

Gradient clipping is not built into this loop. If the loss spikes, compute the global gradient norm and scale every gradient down before the update.

## 20.3 Greedy generation

Generation re-runs the decoder for every new token and keeps the whole prefix. There is no KV cache, so cost grows with the square of the sequence length. The following loop takes the logits at the last position and appends the arg-max token:

```
private static int[] generate(TransformerDecoder decoder, Linear head,
                              int[] prompt, int newTokens) {
    int[] sequence = java.util.Arrays.copyOf(prompt, prompt.length + newTokens);
    int length = prompt.length;

    for (int step = 0; step < newTokens; step++) {
        Tensor hidden = decoder.forward(sequence, 1, length);   // only prefix
        Tensor logits = head.forward(hidden);                   // {1,length,V}
        Tensor flat = Ops.flattenLeading(logits);               // {length,V}

        int lastRow = length - 1;
        Tensor last = Ops.gatherRows(flat, new int[]{lastRow}); // {1,V}
        float[] scores = last.data();

        int best = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[best]) {
                best = i;
            }
        }
        sequence[length++] = best;
    }
    return sequence;
}
```

Because `forward` reads `sequence` for exactly `length` tokens, the unused tail of the array is ignored. For temperature sampling, divide the logits by a temperature `T`, apply `Ops.softmax`, then sample from the resulting distribution; for top-k, keep the `k` largest logits and set the rest to a very negative value before the softmax.

## 20.4 What GPT still needs

| Component | Status | Notes |
| --- | --- | --- |
| Token embedding | Available | `transformers.Embedding` |
| Causal self-attention | Available | `-1e9` mask above the diagonal |
| Decoder stack | Available | `TransformerDecoder` |
| LM head | Buildable | `Linear(D,V)`, separate from the embedding |
| Position embeddings | Missing | Add learned or sinusoidal positions before the blocks |
| BPE tokenizer | Missing | `data.text.Tokenizer` is whitespace-only |
| Sampling utilities | Missing | Temperature, top-k, top-p and repetition penalties |
| KV cache | Missing | Every generation step recomputes the prefix |
| Weight tying | Missing | The head and embedding matrices are separate |
| Dropout | Missing | No dropout inside blocks or embeddings |
| Pretrained weights | Not wired | `SafetensorsIO`/`GgufIO` can read tensors; no GPT name mapping |
| Mixed precision | Missing | Everything is float32 |

## 20.5 Scaling intuition

Parameter count is dominated by `12·L·D²` per block group: attention contributes `4D²`, the feed-forward with `F=4D` contributes `8D²`, and the two layer norms are negligible. Add `V·D` for the embedding and another `V·D` for the head. A tiny model with `D=64, L=3, V=512` has about 0.2 million parameters and trains on a laptop; a GPT-2-small shape (`D=768, L=12, V=50_257`) is roughly 124 million and is not practical with this pure-Java float implementation.

> **This is a teaching implementation.** It favours readable code over throughput. There is no fused attention kernel, no memory-efficient attention, and no distributed training. Use it to understand the shapes, the mask, and the loss; use a production framework for large-scale pretraining.

---

[← 19 · BERT](19-bert.md) · [21 · API reference →](21-api-reference.md)
