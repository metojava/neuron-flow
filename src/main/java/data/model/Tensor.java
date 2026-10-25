package data.model;

import java.util.Arrays;

/**
 * A named float32 tensor stored in row-major order.
 *
 * @param name  tensor name as stored in the file
 * @param shape dimensions, outermost first (row-major)
 * @param data  values; length must equal the product of {@code shape}
 */
public record Tensor(String name, int[] shape, float[] data) {

    public Tensor {
        long n = 1;
        for (int d : shape) {
            if (d < 0) {
                throw new IllegalArgumentException("Tensor " + name + ": negative dimension");
            }
            n *= d;
        }
        if (n != data.length) {
            throw new IllegalArgumentException("Tensor " + name + ": shape "
                    + Arrays.toString(shape) + " needs " + n + " values, got " + data.length);
        }
    }
}
