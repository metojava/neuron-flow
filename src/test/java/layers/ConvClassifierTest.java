package layers;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ConvClassifierTest {

    private static float[] square(int side, int top, int left, int size) {
        float[] image = new float[side * side];
        for (int r = top; r < top + size; r++) {
            for (int c = left; c < left + size; c++) {
                image[r * side + c] = 1.0f;
            }
        }
        return image;
    }

    @Test
    void trainingLowersLossOnRepeatedSample() {
        ConvClassifier net = new ConvClassifier(7L, 28, 10);
        float[] image = square(28, 8, 8, 12);

        net.zeroGrad();
        net.forward(image);
        float initialLoss = net.backward(2);
        for (int step = 0; step < 10; step++) {
            net.applyUpdate(0.01f, 1);
            net.zeroGrad();
            net.forward(image);
            net.backward(2);
        }
        net.forward(image);
        float finalLoss = net.backward(2);

        assertTrue(Float.isFinite(finalLoss));
        assertTrue(finalLoss < initialLoss,
            "Training should lower loss on a repeated sample");
    }

    @Test
    void compactModelHasFewerParametersAndTrains() {
        ConvClassifier compact = ConvClassifier.compact(7L, 28, 10);
        ConvClassifier full = new ConvClassifier(7L, 28, 10);
        assertTrue(compact.parameterCount() < full.parameterCount());

        float[] image = square(28, 13, 13, 2);
        compact.forward(image);
        float before = compact.backward(3);
        compact.applyUpdate(0.01f, 1);
        compact.zeroGrad();
        compact.forward(image);
        assertTrue(compact.backward(3) < before);
    }

    @Test
    void worksForSmallInputsAndCustomClassCounts() {
        ConvClassifier net = new ConvClassifier(3L, 8, 4);
        float[] logits = net.forward(square(8, 2, 2, 4));
        assertEquals(4, logits.length);
        int prediction = net.predict(square(8, 2, 2, 4));
        assertTrue(prediction >= 0 && prediction < 4);
    }

    @Test
    void rejectsInvalidInputsLabelsAndOrder() {
        ConvClassifier net = new ConvClassifier(7L, 28, 10);
        assertThrows(IllegalArgumentException.class,
            () -> net.forward(new float[12]));
        assertThrows(IllegalStateException.class, () -> net.backward(1));
        net.forward(new float[28 * 28]);
        assertThrows(IllegalArgumentException.class, () -> net.backward(10));
        assertThrows(IllegalArgumentException.class,
            () -> new ConvClassifier(7L, 3, 10));
    }
}
