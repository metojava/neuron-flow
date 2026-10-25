package layers;

import org.junit.jupiter.api.Test;
import training.DataParallelTrainer;

import static org.junit.jupiter.api.Assertions.*;

class DenseClassifierTest {

    private static final int[] SIZES = {6, 5, 3};

    private static float[] sample(int s) {
        float[] x = new float[SIZES[0]];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) Math.sin(0.7 * s + 0.4 * i);
        }
        return x;
    }

    /** Loss of one example at the model's current parameters. */
    private static double lossAt(DenseClassifier m, float[] x, int label) {
        m.forward(x);
        m.zeroGrad();
        return m.backward(label);
    }

    @Test
    void gradientsMatchFiniteDifferencesOnEveryParameterKind() {
        DenseClassifier m = new DenseClassifier(3L, SIZES);
        float[] x = sample(1);
        int label = 2;
        double h = 1e-3;

        float[] params = m.getParameters();
        lossAt(m, x, label);
        float[] grads = m.getGradients();

        // Check a spread of indices: first-layer weights, a bias, second-layer weights.
        int[] indices = {0, 7, 29, 30, 33, 40, params.length - 1};
        for (int k : indices) {
            float saved = params[k];
            params[k] = (float) (saved + h);
            m.setParameters(params);
            double up = lossAt(m, x, label);
            params[k] = (float) (saved - h);
            m.setParameters(params);
            double down = lossAt(m, x, label);
            params[k] = saved;
            m.setParameters(params);
            assertEquals((up - down) / (2 * h), grads[k], 5e-3, "param " + k);
        }
    }

    @Test
    void parallelSgdEpochMatchesSerialTraining() {
        int samples = 29;
        float[][] images = new float[samples][];
        int[] labels = new int[samples];
        for (int s = 0; s < samples; s++) {
            images[s] = sample(s);
            labels[s] = s % SIZES[2];
        }
        int[] order = new int[samples];
        for (int i = 0; i < samples; i++) {
            order[i] = (i * 11) % samples;
        }
        int batch = 4;
        float lr = 0.05f;

        DenseClassifier serial = new DenseClassifier(8L, SIZES);
        for (int start = 0; start < samples; start += batch) {
            int end = Math.min(start + batch, samples);
            serial.zeroGrad();
            for (int i = start; i < end; i++) {
                serial.forward(images[order[i]]);
                serial.backward(labels[order[i]]);
            }
            serial.applyUpdate(lr, end - start);
        }

        DenseClassifier parallel = new DenseClassifier(8L, SIZES);
        try (DataParallelTrainer trainer = new DataParallelTrainer(parallel, 3)) {
            trainer.trainEpoch(images, labels, order, samples, batch, lr);
        }

        float[] a = serial.getParameters();
        float[] b = parallel.getParameters();
        for (int k = 0; k < a.length; k++) {
            assertEquals(a[k], b[k], 1e-5f, "parameter " + k);
        }
    }

    @Test
    void adamTrainingLowersLossAndPredictsClasses() {
        DenseClassifier m = new DenseClassifier(4L, SIZES,
            new optim.Adam(0.01f));
        float[][] images = new float[12][];
        int[] labels = new int[12];
        for (int s = 0; s < 12; s++) {
            images[s] = sample(s);
            labels[s] = s % 3;
        }
        int[] order = new int[12];
        for (int i = 0; i < 12; i++) {
            order[i] = i;
        }
        double before = 0.0;
        for (int s = 0; s < 12; s++) {
            before += lossAt(m, images[s], labels[s]);
        }
        try (DataParallelTrainer trainer = new DataParallelTrainer(m, 2)) {
            for (int epoch = 0; epoch < 30; epoch++) {
                trainer.trainEpoch(images, labels, order, 12, 4, 0.01f);
            }
        }
        double after = 0.0;
        for (int s = 0; s < 12; s++) {
            after += lossAt(m, images[s], labels[s]);
        }
        assertTrue(after < before, "Adam training should reduce loss");
        int p = m.predict(images[0]);
        assertTrue(p >= 0 && p < 3);
    }

    @Test
    void rejectsBadConfigurationAndInputs() {
        assertThrows(IllegalArgumentException.class,
            () -> new DenseClassifier(1L, new int[] {4}));
        DenseClassifier m = new DenseClassifier(1L, SIZES);
        assertThrows(IllegalArgumentException.class,
            () -> m.forward(new float[2]));
        assertThrows(IllegalStateException.class,
            () -> new DenseClassifier(1L, SIZES).backward(0));
        m.forward(sample(0));
        assertThrows(IllegalArgumentException.class, () -> m.backward(3));
    }
}
