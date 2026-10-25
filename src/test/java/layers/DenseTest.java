package layers;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DenseTest {

    /** Loss = sum(y * readout), so dLoss/dy = readout. */
    private static double loss(Dense layer, float[] x, float[] readout) {
        float[] y = layer.forward(x);
        double sum = 0.0;
        for (int j = 0; j < y.length; j++) {
            sum += y[j] * readout[j];
        }
        return sum;
    }

    @Test
    void inputAndWeightGradientsMatchFiniteDifferences() {
        Dense layer = new Dense(4, 3, 9L);
        float[] x = {0.5f, -1.2f, 2.0f, 0.1f};
        float[] readout = {0.7f, -0.4f, 1.1f};
        double h = 1e-3;

        layer.forward(x);
        layer.zeroGrad();
        float[] dIn = layer.backward(readout);

        for (int i = 0; i < x.length; i++) {
            float saved = x[i];
            x[i] = (float) (saved + h);
            double up = loss(layer, x, readout);
            x[i] = (float) (saved - h);
            double down = loss(layer, x, readout);
            x[i] = saved;
            assertEquals((up - down) / (2 * h), dIn[i], 2e-3,
                "input grad " + i);
        }

        // One SGD step with lr=1, batch=1 moves W[i][j] by -dLoss/dW[i][j].
        float w00 = layer.weight(0, 0);
        layer.update(1.0f, 1);
        float analytic = w00 - layer.weight(0, 0);
        // dLoss/dW[0][0] = x[0] * readout[0]
        assertEquals(x[0] * readout[0], analytic, 1e-5, "weight grad");
    }

    @Test
    void rejectsWrongInputLengthAndBackwardBeforeForward() {
        Dense layer = new Dense(4, 3, 9L);
        assertThrows(IllegalArgumentException.class,
            () -> layer.forward(new float[3]));
        assertThrows(IllegalStateException.class,
            () -> new Dense(4, 3, 9L).backward(new float[3]));
        assertEquals(4 * 3 + 3, layer.parameterCount());
    }
}
