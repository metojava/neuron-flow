package data;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DataPreparationTest {
    @Test
    void scalerUsesTrainingStatisticsForNewRows() {
        StandardScaler scaler = StandardScaler.fit(new float[][] {{0, 5}, {2, 5}});
        assertArrayEquals(new float[] {-1, 0}, scaler.transform(new float[][] {{0, 5}})[0], 1e-6f);
        assertArrayEquals(new float[] {9, 0}, scaler.transform(new float[][] {{10, 5}})[0], 1e-6f);
        assertArrayEquals(new float[] {10, 5},
            scaler.inverseTransform(scaler.transform(new float[][] {{10, 5}}))[0], 1e-6f);
        assertThrows(IllegalArgumentException.class,
            () -> scaler.transform(new float[][] {{1, 2, 3}}));
    }

    @Test
    void splitKeepsFeaturesPairedWithTargetsAndIsRepeatable() {
        float[][] features = {{0}, {1}, {2}, {3}, {4}, {5}};
        float[][] targets = {{0}, {10}, {20}, {30}, {40}, {50}};
        DatasetSplit.Split first = DatasetSplit.split(features, targets, 0.33, 7L);
        DatasetSplit.Split second = DatasetSplit.split(features, targets, 0.33, 7L);
        assertEquals(4, first.trainFeatures().length);
        assertEquals(2, first.testFeatures().length);
        assertArrayEquals(first.trainFeatures()[0], second.trainFeatures()[0]);
        for (int i = 0; i < first.trainFeatures().length; i++) {
            assertEquals(first.trainFeatures()[i][0] * 10, first.trainTargets()[i][0]);
        }
        for (int i = 0; i < first.testFeatures().length; i++) {
            assertEquals(first.testFeatures()[i][0] * 10, first.testTargets()[i][0]);
        }
        first.trainFeatures()[0][0] = 99;
        assertNotEquals(99, features[0][0]);
    }
}
