package training;

import layers.DenseClassifier;
import layers.ConvClassifier;
import training.TrainableClassifier;

/**
 * Accumulates gradients over multiple mini-batches before applying
 * a single optimizer step. Useful when GPU/CPU memory limits the
 * batch size but a larger effective batch size is desired.
 *
 * <p>Usage:
 * <pre>
 *   GradientAccumulator acc = new GradientAccumulator(model, accumSteps);
 *   for (int i = 0; i &lt; count; i++) {
 *       acc.accumulate(images[order[i]], labels[order[i]]);
 *       if (acc.shouldStep()) {
 *           acc.step(learningRate);
 *       }
 *   }
 *   acc.step(learningRate); // final partial step
 * </pre>
 */
public final class GradientAccumulator {

    private final TrainableClassifier model;
    private final int accumSteps;
    private int count;
    private double totalLoss;

    /**
     * @param model       the model to train
     * @param accumSteps  number of mini-batches to accumulate before stepping
     */
    public GradientAccumulator(TrainableClassifier model, int accumSteps) {
        if (accumSteps < 1) {
            throw new IllegalArgumentException("accumSteps must be >= 1");
        }
        this.model = model;
        this.accumSteps = accumSteps;
        this.count = 0;
        this.totalLoss = 0.0;
    }

    /**
     * Accumulates gradients for one example.
     *
     * @param image flattened input
     * @param label true class index
     * @return the loss for this example
     */
    public double accumulate(float[] image, int label) {
        model.forward(image);
        double loss = model.backward(label);
        totalLoss += loss;
        count++;
        return loss;
    }

    /**
     * Returns true if enough gradients have been accumulated to take
     * an optimizer step.
     */
    public boolean shouldStep() {
        return count >= accumSteps;
    }

    /**
     * Applies the accumulated gradients with the given learning rate
     * and resets the accumulator. The gradients are averaged over the
     * accumulated count.
     *
     * @param learningRate step size
     */
    public void step(float learningRate) {
        if (count == 0) {
            return;
        }
        model.applyUpdate(learningRate, count);
        model.zeroGrad();
        count = 0;
        totalLoss = 0.0;
    }

    /** Returns the mean loss since the last step. */
    public double meanLoss() {
        return count == 0 ? 0.0 : totalLoss / count;
    }

    /** Returns the number of accumulated examples since the last step. */
    public int accumulatedCount() {
        return count;
    }
}
