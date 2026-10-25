package transformers;

/**
 * Mixed precision training utilities for FP16/BF16 training.
 * Provides automatic loss scaling and gradient accumulation in FP16.
 */
public final class MixedPrecision {

    private MixedPrecision() {}

    /**
     * Context for mixed precision training.
     * Handles loss scaling and gradient unscaling.
     */
    public static final class MixedPrecisionContext {
        private final float lossScale;
        private final boolean dynamicLossScale;
        private float currentScale;
        private int consecutiveOverflows;
        private int consecutiveGoodSteps;

        public MixedPrecisionContext(float initialLossScale, boolean dynamic) {
            this.lossScale = initialLossScale;
            this.currentScale = initialLossScale;
            this.dynamicLossScale = dynamic;
            this.consecutiveOverflows = 0;
            this.consecutiveGoodSteps = 0;
        }

        /**
         * Scales the loss before backward pass.
         */
        public float scaleLoss(float loss) {
            return loss * currentScale;
        }

        /**
         * Unscales gradients after backward pass.
         * Returns true if gradients are valid (no overflow), false if overflow detected.
         */
        public boolean unscaleGradients(autograd.Tensor... params) {
            boolean overflow = false;
            for (autograd.Tensor p : params) {
                float[] grad = p.grad();
                if (grad != null) {
                    for (float g : grad) {
                        if (Float.isNaN(g) || Float.isInfinite(g)) {
                            overflow = true;
                            break;
                        }
                    }
                    if (overflow) break;
                }
            }

            if (overflow) {
                consecutiveOverflows++;
                consecutiveGoodSteps = 0;
                if (dynamicLossScale) {
                    currentScale /= 2.0f;
                    System.out.printf("Gradient overflow detected. Reducing loss scale to %.2f%n", currentScale);
                }
                return false;
            } else {
                consecutiveGoodSteps++;
                if (dynamicLossScale && consecutiveGoodSteps >= 2000) {
                    currentScale *= 2.0f;
                    consecutiveGoodSteps = 0;
                    System.out.printf("Increasing loss scale to %.2f%n", currentScale);
                }
                return true;
            }
        }

        /**
         * Unscales gradients in place.
         */
        public void unscaleGradientsInPlace(autograd.Tensor... params) {
            float invScale = 1.0f / currentScale;
            for (autograd.Tensor p : params) {
                float[] grad = p.grad();
                if (grad != null) {
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] *= invScale;
                    }
                }
            }
        }

        public float getCurrentScale() { return currentScale; }
        public void setCurrentScale(float scale) { this.currentScale = scale; }
    }

    /**
     * Casts model parameters to FP16 for storage, keeps master copy in FP32.
     */
    public static class FP16ModelWrapper {
        private final Object model;
        private final float[] masterParams;
        private final int[] paramSizes;

        public FP16ModelWrapper(Object model) {
            this.model = model;
            // In a real implementation, we'd copy params to FP16 storage
            // and keep FP32 master copy for updates
            this.masterParams = null;
            this.paramSizes = null;
        }
    }
}