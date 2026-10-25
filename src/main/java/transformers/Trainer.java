package transformers;

import autograd.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * High-level training loop with gradient accumulation, logging,
 * checkpointing, and learning rate scheduling.
 */
public final class Trainer {

    private final Object model;
    private final AdamW optimizer;
    private final LRScheduler lrScheduler;
    private final int gradientAccumulationSteps;
    private final float maxGradNorm;
    private final int loggingSteps;
    private final int saveSteps;
    private final Path outputDir;

    private int globalStep;
    private int epoch;
    private float runningLoss;
    private int stepsSinceLog;

    public Trainer(Builder builder) {
        this.model = builder.model;
        this.optimizer = builder.optimizer;
        this.lrScheduler = builder.lrScheduler;
        this.gradientAccumulationSteps = builder.gradientAccumulationSteps;
        this.maxGradNorm = builder.maxGradNorm;
        this.loggingSteps = builder.loggingSteps;
        this.saveSteps = builder.saveSteps;
        this.outputDir = builder.outputDir;
        this.globalStep = 0;
        this.epoch = 0;
        this.runningLoss = 0.0f;
        this.stepsSinceLog = 0;
    }

    /**
     * Runs training for the specified number of epochs.
     *
     * @param trainDataProvider supplier that yields batches (inputIds, labels, attentionMask, etc.)
     * @param numEpochs number of epochs to train
     * @param stepFn function that runs a single forward/backward step and returns loss
     */
    public void train(DataProvider trainDataProvider, int numEpochs,
                      TriConsumer<int[], int[], int[]> stepFn) throws IOException {
        
        for (epoch = 0; epoch < numEpochs; epoch++) {
            trainDataProvider.reset();
            
            while (trainDataProvider.hasNext()) {
                // Get batch
                Batch batch = trainDataProvider.next();
                
                // Forward + backward
                autograd.Tensor loss = stepFn.accept(batch.inputIds, batch.labels, batch.attentionMask);
                
                // Accumulate loss
                float lossVal = loss.toFlat()[0];
                runningLoss += lossVal;
                stepsSinceLog++;

                // Gradient accumulation
                if ((globalStep + 1) % gradientAccumulationSteps == 0) {
                    // Clip gradients
                    if (maxGradNorm > 0) {
                        clipGradients(maxGradNorm);
                    }
                    
                    // Optimizer step
                    List<float[]> grads = collectGradients();
                    optimizer.step(grads);
                    optimizer.zeroGrad();
                    
                    // LR scheduler step
                    if (lrScheduler != null) {
                        lrScheduler.step();
                    }
                    
                    globalStep++;
                    
                    // Logging
                    if (globalStep % loggingSteps == 0) {
                        float avgLoss = runningLoss / stepsSinceLog;
                        System.out.printf("Epoch %d | Step %d | Loss: %.4f | LR: %.6f%n",
                            epoch, globalStep, runningLoss / stepsSinceLog, 
                            lrScheduler != null ? lrScheduler.currentLr() : optimizer.lr());
                        runningLoss = 0.0f;
                        stepsSinceLog = 0;
                    }
                    
                    // Checkpoint
                    if (saveSteps > 0 && globalStep % saveSteps == 0) {
                        saveCheckpoint();
                    }
                }
            }
        }
        
        // Final save
        saveCheckpoint();
        System.out.println("Training complete. Final step: " + globalStep);
    }

    private void clipGradients(float maxNorm) {
        // Simple global norm clipping
        float totalNorm = 0.0f;
        for (Tensor p : getParameters()) {
            float[] grad = p.grad();
            if (grad != null) {
                float norm = 0;
                for (float g : grad) totalNorm += g * g;
            }
        }
        totalNorm = (float) Math.sqrt(totalNorm);
        if (totalNorm > maxNorm) {
            float scale = maxGradNorm / totalNorm;
            for (Tensor p : getParameters()) {
                float[] grad = p.grad();
                if (grad != null) {
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] *= scale;
                    }
                }
            }
        }
    }

    private List<autograd.Tensor> getParameters() {
        try {
            java.lang.reflect.Method m = model.getClass().getMethod("parameters");
            @SuppressWarnings("unchecked")
            List<Tensor> params = (List<Tensor>) m.invoke(model);
            return params;
        } catch (Exception e) {
            throw new IllegalStateException("Model must have parameters() method", e);
        }
    }

    private List<float[]> collectGradients() {
        List<float[]> grads = new ArrayList<>();
        for (Tensor p : getParameters()) {
            grads.add(p.grad());
        }
        return grads;
    }

    private void saveCheckpoint() throws IOException {
        if (outputDir == null) return;
        Path dir = outputDir.resolve("checkpoint-" + globalStep);
        ModelSerializer.save(model, getConfig(), dir);
        System.out.println("Saved checkpoint: " + dir);
    }

    private TransformerConfig getConfig() {
        try {
            java.lang.reflect.Method m = model.getClass().getMethod("config");
            return (TransformerConfig) m.invoke(model);
        } catch (Exception e) {
            // Fallback
            return new TransformerConfig();
        }
    }

    // --- Data structures ---

    public static final class Batch {
        public final int[] inputIds;
        public final int[] labels;
        public final int[] attentionMask;

        public Batch(int[] inputIds, int[] labels, int[] attentionMask) {
            this.inputIds = inputIds;
            this.labels = labels;
            this.attentionMask = attentionMask;
        }
    }

    public interface DataProvider {
        boolean hasNext();
        Batch next();
        void reset();
    }

    @FunctionalInterface
    public interface TriConsumer<T, U, V> {
        autograd.Tensor accept(T t, U u, V v);
    }

    // --- Builder ---

    public static final class Builder {
        private Object model;
        private AdamW optimizer;
        private LRScheduler lrScheduler;
        private int gradientAccumulationSteps = 1;
        private float maxGradNorm = 1.0f;
        private int loggingSteps = 100;
        private int saveSteps = 500;
        private Path outputDir;

        public Builder model(Object model) { this.model = model; return this; }
        public Builder optimizer(AdamW optimizer) { this.optimizer = optimizer; return this; }
        public Builder lrScheduler(LRScheduler lrScheduler) { this.lrScheduler = lrScheduler; return this; }
        public Builder gradientAccumulationSteps(int steps) { this.gradientAccumulationSteps = steps; return this; }
        public Builder maxGradNorm(float norm) { this.maxGradNorm = norm; return this; }
        public Builder loggingSteps(int steps) { this.loggingSteps = steps; return this; }
        public Builder saveSteps(int steps) { this.saveSteps = steps; return this; }
        public Builder outputDir(String path) { this.outputDir = Paths.get(path); return this; }

        public Trainer build() {
            if (model == null) throw new IllegalStateException("model required");
            if (optimizer == null) throw new IllegalStateException("optimizer required");
            return new Trainer(this);
        }
    }
}