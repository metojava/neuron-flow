# Encoder-only models: what exists and what is missing

*NeuronFlow does not ship a class called `Bert` or `BertModel`. It ships the encoder backbone and the autograd engine needed to build one. This chapter maps the gap precisely, shows how far you can get with the current API, and lists the pieces a full BERT implementation still needs.*

## 19.1 Status

> **Not implemented end-to-end.** There is no WordPiece tokenizer, no `[CLS]`/`[SEP]` handling, no segment or positional embeddings, no masked-language-model or next-sentence-prediction heads, and no pretrained-weight loader for these classes.

| BERT component | NeuronFlow equivalent | Status |
| --- | --- | --- |
| Token embeddings | `transformers.Embedding` | Available |
| Position embeddings | — | Missing |
| Segment (A/B) embeddings | — | Missing |
| Encoder blocks | `transformers.TransformerBlock` with `useCausal=false` | Available |
| Encoder stack | `transformers.TransformerEncoder` | Available |
| Attention mask for padding | — | Missing |
| MLM head | manual `Linear` stack | Buildable |
| NSP head | manual `Linear` on the pooled vector | Buildable |
| WordPiece tokenizer | `data.text.Tokenizer` (whitespace only) | Not equivalent |
| Pretrained weights | `data.model.SafetensorsIO` / `GgufIO` | Readable, but no name mapping into `transformers.*` |

## 19.2 The encoder backbone you can build today

`TransformerEncoder` already contains the repeated pre-norm block: bidirectional self-attention, residual connections, and a position-wise feed-forward network with GELU. Given token ids of shape `{B,T}` flattened into an `int[]`, it returns contextual states of shape `{B,T,D}`.

```
TransformerEncoder encoder = new TransformerEncoder(
        vocabSize,   // e.g. 30_522 for a BERT vocabulary
        dModel,      // 768 for BERT-base
        numHeads,    // 12
        numLayers,   // 12
        ffnHidden,   // 3072
        42L);

Tensor states = encoder.forward(tokenIds, batchSize, seqLen); // {B,T,768}
```

What is absent is everything around the backbone: positions, segments, masking conventions and the pretraining objectives. A BERT input is not just a token id sequence; it is the sum of token, position and segment embeddings, and the loss is computed only on masked positions.

## 19.3 Adding position embeddings

The cleanest change is to add a learned position table after the token embedding. The current `Embedding` maps ids to rows, so either create a second `Embedding(maxTokens, dModel, seed)` and add its output to the token embeddings, or extend `TransformerEncoder` with a position tensor of shape `{maxT,D}`. Because `Ops.add` broadcasts from the right, the position output must be shaped `{1,T,D}` or `{B,T,D}` to align with the token output.

```
Embedding tokenEmbedding = new Embedding(vocabSize, dModel, 1L);
Embedding positionEmbedding = new Embedding(maxTokens, dModel, 2L);

Tensor tokens = tokenEmbedding.forward(ids, B, T);          // {B,T,D}
int[] positions = new int[B * T];
for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
        positions[b * T + t] = t;
    }
}
Tensor positionsOut = positionEmbedding.forward(positions, B, T);
Tensor input = Ops.add(tokens, positionsOut);               // {B,T,D}
```

Segment embeddings work the same way with ids `0` and `1`. BERT also uses learned absolute positions, which is why an `Embedding` is a reasonable fit.

## 19.4 A masked-language-model head

An MLM head projects each contextual vector back to the vocabulary and computes cross-entropy only at the positions that were masked. With the current API you can do this by flattening to `{B*T,D}`, projecting to `{B*T,V}`, selecting the masked rows with `Ops.gatherRows`, and passing only those rows to `Ops.crossEntropy`.

```
Linear transform = new Linear(dModel, dModel, 10L);
LayerNorm norm = new LayerNorm(dModel);
Linear decoder = new Linear(dModel, vocabSize, 11L);

Tensor hidden = encoder.forward(maskedIds, B, T);           // {B,T,D}
Tensor flat = Ops.flattenLeading(hidden);                   // {B*T,D}
Tensor projected = Ops.gelu(transform.forward(flat));
projected = norm.forward(projected);
Tensor logits = decoder.forward(projected);                 // {B*T,V}

// positions where the input token was replaced by [MASK]
int[] maskedRows = /* indices in [0, B*T) */;
int[] maskedTargets = /* original token id for each masked row */;

Tensor selected = Ops.gatherRows(logits, maskedRows);       // {M,V}
Tensor loss = Ops.crossEntropy(selected, maskedTargets);    // scalar mean
```

BERT ties the input embedding matrix and the output projection in the original model. Here `Linear` always creates its own weight, so the two matrices are separate unless you add a custom tied head.

## 19.5 Fine-tuning

For sentence classification, BERT uses the final `[CLS]` state (the first token of each sequence) as the sentence representation. The current API has no pooler, but the data is there: after `encoder.forward`, the first position of every sequence is at flat offset `b*T*D`. You can copy those rows into a `{B,D}` tensor, apply a small `Linear(D, numLabels)` head, and train with `Ops.crossEntropy`. Freezing the backbone is not directly supported: there is no `requiresGrad=false` switch on an existing parameter, so only update the head's gradients in your optimiser loop.

## 19.6 What a full implementation needs

1. **WordPiece tokenizer** with vocabulary loading, greedy longest-match subwords, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` and `[PAD]`.
2. **Input assembly** that sums token, position and segment embeddings and records the attention mask.
3. **Padding mask** passed into `MultiHeadAttention` so padded keys receive `-1e9` before softmax.
4. **Pretraining objectives**: 15% masked-token selection with 80/10/10 replacement, plus the next-sentence-prediction pair classifier if you want the original recipe.
5. **Weight loading**: a name-mapping layer from safetensors/GGUF tensors to `transformers.*` parameters. `layers.WeightImporter` only targets the `layers` classes.
6. **Training utilities**: learning-rate warm-up and decay, gradient clipping, mixed precision if desired, and checkpointing.
7. **Evaluation**: masked-token accuracy, perplexity, and GLUE-style task metrics.

> **Use the source as a specification.** The transformer blocks are deliberately small and readable. Extending `TransformerEncoder` with a position table and a pooling output is a good first project; it teaches more than importing a pretrained model.

---

[← 18 · Transformers](18-transformers.md) · [20 · GPT →](20-gpt.md)
