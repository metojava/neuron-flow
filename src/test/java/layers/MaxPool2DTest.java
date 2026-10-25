package layers;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MaxPool2DTest {

    @Test
    void backwardRoutesGradientToWindowMaximum() {
        MaxPool2D pool = MaxPool2D.standard();
        float[][][] in = new float[1][4][4];
        in[0][1][2] = 5.0f;   // max of top-right window
        in[0][3][0] = 7.0f;   // max of bottom-left window

        float[][][] out = pool.forward(in);
        assertEquals(5.0f, out[0][0][1], 1e-6);
        assertEquals(7.0f, out[0][1][0], 1e-6);

        float[][][] dOut = new float[1][2][2];
        dOut[0][0][1] = 2.0f;
        dOut[0][1][0] = 3.0f;
        float[][][] dIn = pool.backward(dOut);

        assertEquals(2.0f, dIn[0][1][2], 1e-6);
        assertEquals(3.0f, dIn[0][3][0], 1e-6);
        float total = 0.0f;
        for (float[] row : dIn[0]) {
            for (float v : row) {
                total += v;
            }
        }
        assertEquals(5.0f, total, 1e-6, "gradient must only land on maxima");
    }

    @Test
    void backwardBeforeForwardIsRejected() {
        assertThrows(IllegalStateException.class,
            () -> MaxPool2D.standard().backward(new float[1][2][2]));
    }
}
