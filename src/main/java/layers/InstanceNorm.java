package layers;

/**
 * Instance normalization: normalizes each sample and each channel
 * independently using its own mean and variance. Common in style
 * transfer and GANs.
 */
public final class InstanceNorm {

    private final int channels;
    private final float[] gamma;
    private final float[] beta;
    private final float epsilon;

    /**
     * @param channels number of channels (feature maps)
     * @param epsilon  small constant for numerical stability
     */
    public InstanceNorm(int channels, float epsilon) {
        if (channels <= 0) {
            throw new IllegalArgumentException("channels must be positive");
        }
        this.channels = channels;
        this.gamma = new float[channels];
        this.beta = new float[channels];
        for (int i = 0; i < channels; i++) {
            gamma[i] = 1.0f;
        }
        this.epsilon = epsilon;
    }

    /** Default epsilon of 1e-5. */
    public InstanceNorm(int channels) {
        this(channels, 1e-5f);
    }

    /**
     * Normalizes a single channel's feature map.
     *
     * @param channel [height][width] feature map
     * @return normalized map
     */
    public float[][] forward(float[][] channel, int channelIndex) {
        int h = channel.length;
        int w = channel[0].length;
        float mean = 0.0f;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                mean += channel[i][j];
            }
        }
        mean /= (h * w);

        float var = 0.0f;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                float d = channel[i][j] - mean;
                var += d * d;
            }
        }
        var /= (h * w);

        float std = (float) Math.sqrt(var + epsilon);
        float[][] out = new float[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                out[i][j] = gamma[channelIndex] * (channel[i][j] - mean) / std
                    + beta[channelIndex];
            }
        }
        return out;
    }

    /**
     * Normalizes a multi-channel tensor [C][H][W].
     *
     * @param input [channels][height][width]
     * @return normalized tensor
     */
    public float[][][] forward(float[][][] input) {
        int channels = input.length;
        float[][][] out = new float[channels][][];
        for (int c = 0; c < channels; c++) {
            out[c] = forward(input[c], c);
        }
        return out;
    }
}
