package optim;

/**
 * Utility for clipping gradients before an optimizer step.
 *
 * <p>Two modes are supported:
 * <ul>
 *   <li><b>By value:</b> each element is clamped to [-max, max].</li>
 *   <li><b>By norm:</b> if the L2 norm of the gradient vector exceeds
 *       {@code maxNorm}, the whole vector is scaled down so its norm
 *       equals {@code maxNorm}.</li>
 * </ul>
 */
public final class GradientClipper {

    private GradientClipper() {
    }

    /**
     * Clips every element of {@code gradients} to [-maxAbs, maxAbs] in
     * place.
     *
     * @param gradients gradient vector, modified in place
     * @param maxAbs    maximum absolute value per element, must be &gt; 0
     * @throws IllegalArgumentException on null or non-positive maxAbs
     */
    public static void clipByValue(float[] gradients, float maxAbs) {
        if (gradients == null) {
            throw new IllegalArgumentException("gradients must not be null");
        }
        if (!(maxAbs > 0.0f)) {
            throw new IllegalArgumentException("maxAbs must be positive");
        }
        for (int i = 0; i < gradients.length; i++) {
            if (gradients[i] > maxAbs) {
                gradients[i] = maxAbs;
            } else if (gradients[i] < -maxAbs) {
                gradients[i] = -maxAbs;
            }
        }
    }

    /**
     * Clips the gradient vector so its L2 norm does not exceed
     * {@code maxNorm}. If the norm is already within the limit the
     * vector is unchanged.
     *
     * @param gradients gradient vector, modified in place
     * @param maxNorm   maximum L2 norm, must be &gt; 0
     * @throws IllegalArgumentException on null or non-positive maxNorm
     */
    public static void clipByNorm(float[] gradients, float maxNorm) {
        if (gradients == null) {
            throw new IllegalArgumentException("gradients must not be null");
        }
        if (!(maxNorm > 0.0f)) {
            throw new IllegalArgumentException("maxNorm must be positive");
        }
        double sumSq = 0.0;
        for (float g : gradients) {
            sumSq += (double) g * g;
        }
        double norm = Math.sqrt(sumSq);
        if (norm > maxNorm) {
            float scale = (float) (maxNorm / norm);
            for (int i = 0; i < gradients.length; i++) {
                gradients[i] *= scale;
            }
        }
    }

    /**
     * Returns the L2 norm of a gradient vector.
     *
     * @param gradients gradient vector
     * @return the L2 norm
     * @throws IllegalArgumentException on null
     */
    public static float norm(float[] gradients) {
        if (gradients == null) {
            throw new IllegalArgumentException("gradients must not be null");
        }
        double sumSq = 0.0;
        for (float g : gradients) {
            sumSq += (double) g * g;
        }
        return (float) Math.sqrt(sumSq);
    }
}
