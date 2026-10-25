package activations;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ReluHelpersTest {

    @Test
    void vectorReluLeavesInputUnchanged() {
        float[] x = {-2.0f, 0.0f, 3.5f};
        assertArrayEquals(new float[] {0.0f, 0.0f, 3.5f}, Activations.relu(x));
        assertArrayEquals(new float[] {-2.0f, 0.0f, 3.5f}, x);
    }

    @Test
    void vectorReluBackwardUsesPreActivationSign() {
        float[] pre = {-1.0f, 0.0f, 2.0f};
        float[] dOut = {5.0f, 6.0f, 7.0f};
        assertArrayEquals(new float[] {0.0f, 0.0f, 7.0f},
            Activations.reluBackward(dOut, pre));
    }

    @Test
    void tensorReluAndBackwardMatchElementwiseRule() {
        float[][][] x = {{{-1.0f, 2.0f}, {0.5f, -0.5f}}};
        float[][][] y = Activations.relu(x);
        assertArrayEquals(new float[] {0.0f, 2.0f}, y[0][0]);
        assertArrayEquals(new float[] {0.5f, 0.0f}, y[0][1]);

        float[][][] dOut = {{{1.0f, 1.0f}, {1.0f, 1.0f}}};
        float[][][] d = Activations.reluBackward(dOut, x);
        assertArrayEquals(new float[] {0.0f, 1.0f}, d[0][0]);
        assertArrayEquals(new float[] {1.0f, 0.0f}, d[0][1]);
    }
}
