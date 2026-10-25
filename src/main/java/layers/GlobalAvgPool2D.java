package layers;

/**
 * Global average pooling: collapses each feature map to a single value
 * (the mean over all spatial positions). Output is [channels].
 * Commonly used before the dense head of a CNN to replace Flatten.
 */
public final class GlobalAvgPool2D {

    private GlobalAvgPool2D() {
    }

    /** Pools a multi-channel tensor [C][H][W] to [C]. */
    public static float[] forward(float[][][] input) {
        int channels = input.length;
        int h = input[0].length;
        int w = input[0][0].length;
        float[] out = new float[channels];
        float inv = 1.0f / (h * w);

        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    sum += input[c][i][j];
                }
            }
            out[c] = sum * inv;
        }
        return out;
    }

    /**
     * Backward: distributes each output gradient evenly over all
     * spatial positions of the corresponding channel.
     *
     * @param dOut gradient w.r.t. output, length channels
     * @param h    input height
     * @param w    input width
     * @return gradient w.r.t. input, [channels][h][w]
     */
    public static float[][][] backward(float[] dOut, int h, int w) {
        int channels = dOut.length;
        float[][][] dIn = new float[channels][h][w];
        float inv = 1.0f / (h * w);

        for (int c = 0; c < channels; c++) {
            float g = dOut[c] * inv;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    dIn[c][i][j] = g;
                }
            }
        }
        return dIn;
    }
}
