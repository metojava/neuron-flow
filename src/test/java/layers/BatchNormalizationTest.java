package layers;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BatchNormalizationTest {
    @Test
    void inferenceUsesSeparateStatisticsForEachFeature() {
        BatchNormalization normalization = BatchNormalization.of();
        float[][] training = {{0, 100}, {2, 200}};
        normalization.normalize(training);
        float[] sample = {1, 150};
        normalization.normalizeInference(sample);
        assertArrayEquals(new float[] {0, 0}, sample, 1e-5f);
    }

    @Test
    void rejectsInferenceBeforeFitting() {
        assertThrows(IllegalArgumentException.class,
            () -> BatchNormalization.of().normalizeInference(new float[] {1}));
    }
}
