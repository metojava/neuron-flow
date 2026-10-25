package layers;

/**
 * Flatten layer: reshapes a multi-dimensional input (e.g. an
 * image [H][W][C]) into a 1-D vector of length H*W*C. Purely a
 * reshape — no parameters, no computation. Sits between conv
 * stacks and dense layers.
 */
public class Flatten {

    private Flatten() {
    }

    /**
     * Flattens a 2-D spatial array (e.g. a single-channel
     * feature map) into a 1-D vector, row-major.
     *
     * @param map [height][width] feature map
     * @return flat vector of length height * width
     */
    public static float[] flatten(float[][] map) {
        if (map == null || map.length == 0) {
            throw new IllegalArgumentException(
                "Map must be nonempty");
        }

        int h = map.length;
        int w = map[0].length;
        float[] out = new float[h * w];
        int k = 0;

        for (int i = 0; i < h; i++) {
            if (map[i].length != w) {
                throw new IllegalArgumentException(
                    "Ragged feature map");
            }
            for (int j = 0; j < w; j++) {
                out[k++] = map[i][j];
            }
        }

        return out;
    }

    /**
     * Flattens a 3-D tensor (height x width x channels) into a
     * 1-D vector, channel-last order.
     *
     * @param tensor [height][width][channels]
     * @return flat vector of length height * width * channels
     */
    public static float[] flatten(float[][][] tensor) {
        if (tensor == null || tensor.length == 0) {
            throw new IllegalArgumentException(
                "Tensor must be nonempty");
        }

        int h = tensor.length;
        int w = tensor[0].length;
        int c = tensor[0][0].length;
        float[] out = new float[h * w * c];
        int k = 0;

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                for (int ch = 0; ch < c; ch++) {
                    out[k++] = tensor[i][j][ch];
                }
            }
        }

        return out;
    }

    /**
     * Reshapes a flat vector back into a 2-D map (inverse of
     * the 2-D flatten).
     *
     * @param flat     flattened vector
     * @param height   target height
     * @param width    target width
     * @return [height][width] map
     */
    public static float[][] unflatten(float[] flat,
                                      int height, int width) {
        if (flat.length != height * width) {
            throw new IllegalArgumentException(
                "Length " + flat.length + " != "
                + height + " * " + width);
        }

        float[][] out = new float[height][width];
        int k = 0;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                out[i][j] = flat[k++];
            }
        }

        return out;
    }

    /**
     * Flattens a channel-first tensor [channels][height][width], the
     * layout produced by {@link Conv2D} and {@link MaxPool2D}, into a
     * 1-D vector in channel-major order.
     *
     * @param chw [channels][height][width] tensor
     * @return flat vector of length channels * height * width
     */
    public static float[] flattenChannelsFirst(float[][][] chw) {
        int c = chw.length;
        int h = chw[0].length;
        int w = chw[0][0].length;
        float[] out = new float[c * h * w];
        int k = 0;

        for (int ch = 0; ch < c; ch++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    out[k++] = chw[ch][i][j];
                }
            }
        }

        return out;
    }

    /**
     * Inverse of {@link #flattenChannelsFirst}.
     *
     * @param flat     vector of length channels * height * width
     * @param channels target channel count
     * @param height   target height
     * @param width    target width
     * @return [channels][height][width] tensor
     */
    public static float[][][] unflattenChannelsFirst(float[] flat,
                                                     int channels,
                                                     int height,
                                                     int width) {
        if (flat.length != channels * height * width) {
            throw new IllegalArgumentException(
                "Length " + flat.length + " != "
                + channels + " * " + height + " * " + width);
        }

        float[][][] out = new float[channels][height][width];
        int k = 0;

        for (int ch = 0; ch < channels; ch++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    out[ch][i][j] = flat[k++];
                }
            }
        }

        return out;
    }
}
