# NeuronFlow Transformer Examples

This document provides examples for using the NeuronFlow transformer library.

## Table of Contents
1. [Quick Start](#quick-start)
2. [BERT Model](#bert-model)
3. [GPT Model](#gpt-model)
4. [Training](#training)
5. [Generation](#generation)
5. [Serialization](#serialization)
6. [Tokenizer](#tokenizer)

## Quick Start

```java
import transformers.*;
import autograd.Tensor;

// Create a small GPT model
GPTConfig config = GPTConfig.gpt2Small();
config.hiddenSize = 256;
config.numHiddenLayers = 4;
config.numAttentionHeads = 4;
config.intermediateSize = 1024;
config.vocabSize = 50257;
config.maxPositionEmbeddings = 1024;

GPTLMHeadModel model = new GPTLMHeadModel(config, 42L);

// Generate text
int[] prompt = tokenizer.encode("Hello world");
int[] generated = model.generate(prompt, 50, 0.8f, 40, 0.9f, 1.1f, new int[]{50256}, 123L);
String text = tokenizer.decode(generated);
```

## BERT Model

### Basic BERT Model
```java
BertConfig config = BertConfig.bertBaseUncased();
config.hiddenSize = 768;
config.numHiddenLayers = 12;
config.numAttentionHeads = 12;
config.intermediateSize = 3072;
config.vocabSize = 30522;
config.maxPositionEmbeddings = 512;

BertModel bert = new BertModel(config, 42L);

// Forward pass
int[] inputIds = tokenizer.encode("Hello world");
int[] segmentIds = new int[inputIds.length]; // single segment
int[] attentionMask = new int[inputIds.length]; // 1 for real tokens, 0 for padding
Arrays.fill(attentionMask, 1);

BertModel.BertOutput output = bert.forward(inputIds, segmentIds, attentionMask);
Tensor lastHidden = output.lastHiddenState;  // [B, T, D]
Tensor pooled = output.pooledOutput;         // [B, D]
```

### Masked Language Modeling
```java
BertForMaskedLM mlm = new BertForMaskedLM(config, 42L);

// Prepare masked input
int[] inputIds = tokenizer.encode("The [MASK] is blue");
MLMMasker.MaskedResult masked = new MLMMasker(config.maskTokenId, config.vocabSize, 42L).mask(inputIds);

BertForMaskedLM.MaskedLMOutput output = mlm.forward(
    masked.inputIds, null, null, masked.labels);

Tensor loss = output.loss;  // MLM loss
```

### Sequence Classification
```java
BertForSequenceClassification classifier = new BertForSequenceClassification(config, 2, 42L); // binary

BertForSequenceClassification.SequenceClassificationOutput output = 
    classifier.forward(inputIds, segmentIds, attentionMask, labels);

Tensor logits = output.logits;  // [B, numLabels]
Tensor loss = output.loss;      // cross-entropy loss
```

## GPT Model

### Basic GPT Model
```java
GPTConfig config = GPTConfig.gpt2Small();
config.hiddenSize = 768;
config.numHiddenLayers = 12;
config.numAttentionHeads = 12;
config.intermediateSize = 3072;
config.vocabSize = 50257;
config.maxPositionEmbeddings = 1024;

GPTLMHeadModel gpt = new GPTLMHeadModel(config, 42L);

// Forward pass
int[] inputIds = tokenizer.encode("The quick brown fox");
Tensor logits = gpt.forward(inputIds, 1, inputIds.length);
```

### Text Generation
```java
// Greedy decoding
int[] generated = gpt.generate(promptIds, 100, 0.0f, 0, 0.0f, new int[]{50256}, 42L);

// Sampling with temperature
int[] generated = gpt.generate(promptIds, 100, 0.8f, 40, 0.9f, 1.1f, new int[]{50256}, 123L);

// Beam search
BeamSearchDecoder decoder = new BeamSearchDecoder(gptModel, 5);
int[] generated = decoder.generate(promptIds, 50, 0.6f);
```

## Training

### AdamW Optimizer with LR Scheduler
```java
// Get model parameters
List<Tensor> params = model.parameters();

// Create optimizer with weight decay
AdamW optimizer = new AdamW(params, 3e-4f, 0.9f, 0.999f, 1e-8f, 0.01f);

// LR scheduler with warmup
LRScheduler scheduler = new LRScheduler(optimizer, 1000, 100000, 0.1f);

// Training loop
for (int epoch = 0; epoch < numEpochs; epoch++) {
    for (Batch batch : dataLoader) {
        // Forward + backward
        Tensor loss = model.forward(batch.inputIds, batch.labels, batch.attentionMask);
        
        // Accumulate gradients
        loss.backward();
        
        if ((step + 1) % gradAccumSteps == 0) {
            // Clip gradients
            clipGradients(params, 1.0f);
            
            // Optimizer step
            optimizer.step(collectGradients(params));
            optimizer.zeroGrad();
            
            // LR scheduler step
            scheduler.step();
        }
    }
}
```

### Using Trainer
```java
Trainer trainer = new Trainer.Builder()
    .model(model)
    .optimizer(optimizer)
    .lrScheduler(scheduler)
    .gradientAccumulationSteps(4)
    .maxGradNorm(1.0f)
    .loggingSteps(100)
    .saveSteps(500)
    .outputDir("checkpoints")
    .build();

trainer.train(dataProvider, numEpochs, (inputIds, labels, attentionMask) -> {
    return model.forward(inputIds, labels, attentionMask);
});
```

## Serialization

### Save/Load Model
```java
// Save
GPTConfig config = model.config;
ModelSerializer.save(model, config, Paths.get("model_dir"));

// Load
GPTLMHeadModel loaded = (GPTLMHeadModel) ModelSerializer.load(
    GPTLMHeadModel.class, GPTConfig.class, Paths.get("model_dir"));
```

## Tokenizer

### Basic Usage
```java
Tokenizer tokenizer = Tokenizer.fromTiktoken("vocab.json", "merges.txt");

// Encode
int[] ids = tokenizer.encode("Hello world");

// Decode
String text = tokenizer.decode(ids);
```

### Special Tokens
```java
Tokenizer tok = new Tokenizer.Builder()
    .vocab(vocab)
    .merges(merges)
    .build();

tok = tok.addSpecialTokens(Map.of(
    Tokenizer.PAD_TOKEN, 0,
    Tokenizer.UNK_TOKEN, 1,
    Tokenizer.CLS_TOKEN, 2,
    Tokenizer.SEP_TOKEN, 3,
    Tokenizer.MASK_TOKEN, 3,
    Tokenizer.BOS_TOKEN, 4,
    Tokenizer.EOS_TOKEN, 5
));
```

### Encode Plus (with padding/truncation)
```java
Map<String, int[]> result = tok.encodePlus(
    "Hello world", true,  // addSpecialTokens
    128,                  // maxLength
    true,                 // padding
    true                  // truncation
);

int[] inputIds = result.get("input_ids");
int[] attentionMask = result.get("attention_mask");
int[] tokenTypeIds = result.get("token_type_ids");
```

### Batch Encoding
```java
Map<String, int[][]> batch = tok.batchEncodePlus(
    List.of("Hello world", "Goodbye world"),
    true, 128, true, true
);

int[][] inputIds = batch.get("input_ids");
int[][] attentionMask = batch.get("attention_mask");
```

## Mixed Precision Training

```java
MixedPrecisionContext mp = new MixedPrecisionContext(65536.0f, true);

// Forward pass with loss scaling
float scaledLoss = mp.scaleLoss(loss);
scaledLoss.backward();

// Unscale gradients
if (mp.unscaleGradients(params)) {
    optimizer.step(grads);
    optimizer.zeroGrad();
} else {
    // Skip step due to overflow
}
```

## Distributed Training (DDP)

```java
DDPConfig config = new DDPConfig(
    4,          // worldSize
    0,          // rank
    "localhost", // masterAddr
    29500,      // masterPort
    25          // bucketSize MB
);

DistributedDataParallel ddp = new DistributedDataParallel(model, config);

for (Batch batch : dataLoader) {
    Tensor loss = model.forward(batch.inputIds, batch.labels, batch.attentionMask);
    loss.backward();
    
    ddp.syncGradients();  // All-reduce gradients
    
    optimizer.step(grads);
    optimizer.zeroGrad();
    
    ddp.readyForNextStep();
}
```

## Perplexity Evaluation

```java
PerplexityEvaluator evaluator = new PerplexityEvaluator(gptModel);
PerplexityResult result = evaluator.evaluate(dataProvider);

System.out.println(result); // Perplexity: 15.23 | Avg NLL: 2.72 | Tokens: 50000 | Batches: 100
```

## Mixed Precision Training

```java
MixedPrecisionContext mp = new MixedPrecisionContext(65536.0f, true);

// Forward with scaled loss
float scaledLoss = mp.scaleLoss(loss);
scaledLoss.backward();

// Unscale and check for overflow
if (mp.unscaleGradients(params)) {
    // Valid gradients - update
    optimizer.step(grads);
    optimizer.zeroGrad();
} else {
    // Overflow - skip step, reduce scale
}
```

## Beam Search Decoding

```java
BeamSearchDecoder decoder = new BeamSearchDecoder(gptModel, 5);
int[] generated = decoder.generate(promptIds, 50, 0.6f); // lengthPenalty=0.6
```

## Save/Load Model with Serialization

```java
// Save
ModelSerializer.save(model, config, Paths.get("model_dir"));

// Load
TransformerDecoder loaded = (TransformerDecoder) ModelSerializer.load(
    TransformerDecoder.class, TransformerConfig.class, Paths.get("model_dir"));
```

## Safetensors Weight Loading

```java
// Load HF weights from safetensors
HFWeightLoader.loadWeights(model, Paths.get("model.safetensors"), "gpt2");
```

## Mixed Precision Training

```java
MixedPrecision.MixedPrecisionContext mp = new MixedPrecision.MixedPrecisionContext(
    65536.0f, true  // dynamic loss scaling
);

// Scale loss
float scaledLoss = mp.scaleLoss(loss);
scaledLoss.backward();

// Unscale and check for overflow
if (mp.unscaleGradients(params)) {
    optimizer.step(grads);
    optimizer.zeroGrad();
} else {
    // Overflow detected - skip step
}
```