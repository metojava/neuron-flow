package transformers;

/**
 * Learning rate scheduler with linear warmup and cosine decay.
 * Matches Hugging Face / PyTorch behavior.
 */
public final class LRScheduler {

    private final AdamW optimizer;
    private final int warmupSteps;
    private final int totalSteps;
    private final float minLrRatio;
    private int currentStep;

    public LRScheduler(AdamW optimizer, int warmupSteps, int totalSteps, float minLrRatio) {
        this.optimizer = optimizer;
        this.warmupSteps = warmupSteps;
        this.totalSteps = totalSteps;
        this.minLrRatio = minLrRatio;
        this.currentStep = 0;
    }

    /**
     * Steps the scheduler and updates optimizer learning rate.
     * Call after optimizer.step().
     */
    public void step() {
        currentStep++;
        float lr = computeLr();
        optimizer.setLr(lr);
    }

    private float computeLr() {
        float baseLr = optimizer.lr();
        if (currentStep < warmupSteps) {
            // Linear warmup
            return baseLr * currentStep / Math.max(1, warmupSteps);
        } else {
            // Cosine decay
            float progress = (currentStep - warmupSteps) / (float) Math.max(1, totalSteps - warmupSteps);
            progress = Math.min(1.0f, Math.max(0.0f, progress));
            float cosine = (float) (0.5 * (1 + Math.cos(Math.PI * progress)));
            return baseLr * (minLrRatio + (1 - minLrRatio) * cosine);
        }
    }

    public int currentStep() { return currentStep; }
    public float currentLr() { return optimizer.lr(); }
    public void setStep(int step) { this.currentStep = step; }
}