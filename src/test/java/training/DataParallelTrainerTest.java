package training;

import layers.ConvClassifier;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DataParallelTrainerTest {

    @Test
    void parallelEpochMatchesSerialMiniBatchTraining() {
        int side = 8;
        int classes = 3;
        int samples = 37;   // not a multiple of the batch or thread count
        int batch = 8;
        float lr = 0.05f;

        float[][] images = new float[samples][side * side];
        int[] labels = new int[samples];
        for (int s = 0; s < samples; s++) {
            labels[s] = s % classes;
            for (int p = 0; p < side * side; p++) {
                images[s][p] = (float) Math.sin(0.31 * s + 0.17 * p);
            }
        }
        int[] order = new int[samples];
        for (int i = 0; i < samples; i++) {
            order[i] = (i * 7) % samples;   // a fixed permutation
        }

        // Serial reference.
        ConvClassifier serial = ConvClassifier.compact(5L, side, classes);
        for (int start = 0; start < samples; start += batch) {
            int end = Math.min(start + batch, samples);
            serial.zeroGrad();
            for (int i = start; i < end; i++) {
                serial.forward(images[order[i]]);
                serial.backward(labels[order[i]]);
            }
            serial.applyUpdate(lr, end - start);
        }

        // Parallel run from the same initial weights.
        ConvClassifier parallel = ConvClassifier.compact(5L, side, classes);
        try (DataParallelTrainer trainer = new DataParallelTrainer(parallel, 3)) {
            trainer.trainEpoch(images, labels, order, samples, batch, lr);
        }

        float[] a = serial.getParameters();
        float[] b = parallel.getParameters();
        assertEquals(a.length, b.length);
        for (int k = 0; k < a.length; k++) {
            assertEquals(a[k], b[k], 1e-4f, "parameter " + k);
        }
    }

    @Test
    void rejectsNonPositiveThreadCount() {
        ConvClassifier m = ConvClassifier.compact(1L, 8, 2);
        assertThrows(IllegalArgumentException.class,
            () -> new DataParallelTrainer(m, 0));
    }
}
