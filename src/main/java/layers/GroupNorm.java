package layers;

/**
 * Group normalization: divides channels into groups and normalizes
 * within each group. Works well with small batch sizes.
 */
public final class GroupNorm {

    private final int numGroups;
    private final int channels;
    private final float[] gamma;
    private final float[] beta;
    private final float epsilon;

    /**
     * @param numGroups number of groups to divide channels into
     * @param channels  total number of channels (must be divisible by numGroups)
     * @param epsilon   small constant for numerical stability
     */
    public GroupNorm(int numGroups, int channels, float epsilon) {
        if (channels % numGroups != 0) {
            throw new IllegalArgumentException(
                "channels must be divisible by numGroups");
        }
        this.numGroups = numGroups;
        this.channels = channels;
        this.gamma = new float[channels];
        this.beta = new float[channels];
        for (int i = 0; i < channels; i++) {
            gamma[i] = 1.0f;
        }
        this.epsilon = epsilon;
    }

    /** Default epsilon of 1e-5. */
    public GroupNorm(int numGroups, int channels) {
        this(numGroups, channels, 1e-5f);
    }

    /**
     * Normalizes a multi-channel tensor [C][H][W].
     *
     * @param input [channels][height][width]
     * @return normalized tensor
     */
    public float[][][] forward(float[][][] input) {
        int c = input.length;
        int h = input[0].length;
        int w = input[0][0].length;
        int channelsPerGroup = c / numGroups;
        float[][][] out = new float[c][h][w];

        for (int g = 0; g < numGroups; g++) {
            int start = g * channelsPerGroup;
            int end = start + channelsPerGroup;

            // Compute mean and variance over the group
            float mean = 0.0f;
            int count = 0;
            for (int ch = start; ch < end; ch++) {
                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        mean += input[ch][i][j];
                        count++;
                    }
                }
            }
            mean /= count;

            float var = 0.0f;
            for (int ch = start; ch < end; ch++) {
                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        float d = input[ch][i][j] - mean;
                        var += d * d;
                    }
                }
            }
            var /= count;

            float std = (float) Math.sqrt(var + epsilon);
            for (int ch = start; ch < end; ch++) {
                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        out[ch][i][j] = gamma[ch] * (input[ch][i][j] - mean) / std
                            + beta[ch];
                    }
                }
            }
        }
        return out;
    }
}
