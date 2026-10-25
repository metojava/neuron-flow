package evaluation;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;
import javax.imageio.ImageIO;

/**
 * Renders a [0,1] float image as a base64-encoded PNG data URI
 * that can be embedded directly in an HTML img tag.
 * No external libraries - uses BufferedImage + ImageIO.
 */
public final class ImageEncoder {

    private ImageEncoder() {
    }

    /**
     * Converts a flattened grayscale image to a base64 PNG data
     * URI, upscaled by an integer factor for visibility.
     *
     * @param pixels 784 floats in [0, 1] (28x28 image)
     * @param scale  integer upscale factor (e.g. 2 -> 56x56)
     * @return a data URI string: data:image/png;base64,...
     * @throws IOException if PNG encoding fails
     */
    public static String toDataUri(float[] pixels, int scale)
            throws IOException {
        int srcSize = (int) Math.round(
            Math.sqrt(pixels.length));
        int dstSize = srcSize * scale;

        BufferedImage img = new BufferedImage(
            dstSize, dstSize, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < srcSize; y++) {
            for (int x = 0; x < srcSize; x++) {
                float v = pixels[y * srcSize + x];
                int gray = Math.round(
                    Math.max(0.0f, Math.min(1.0f, v)) * 255.0f);
                int rgb = (gray << 16) | (gray << 8) | gray;

                for (int dy = 0; dy < scale; dy++) {
                    for (int dx = 0; dx < scale; dx++) {
                        img.setRGB(x * scale + dx,
                                   y * scale + dy, rgb);
                    }
                }
            }
        }

        ByteArrayOutputStream baos =
            new ByteArrayOutputStream();
        ImageIO.write(img, "png", baos);

        return "data:image/png;base64,"
            + Base64.getEncoder().encodeToString(
                baos.toByteArray());
    }

    /**
     * Converts a flattened grayscale image to a base64 PNG data
     * URI at the default scale of 2 (56x56 output).
     *
     * @param pixels 784 floats in [0, 1]
     * @return a data URI string
     * @throws IOException if PNG encoding fails
     */
    public static String toDataUri(float[] pixels)
            throws IOException {
        return toDataUri(pixels, 2);
    }
}
