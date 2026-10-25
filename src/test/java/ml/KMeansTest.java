package ml;

import ml.model.KMeans;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class KMeansTest {

    @Test
    void recoversThreeWellSeparatedClusters() {
        double[][] points = new double[30][2];
        int[] expected = new int[30];
        for (int i = 0; i < 30; i++) {
            int cluster = i / 10;
            expected[i] = cluster;
            points[i][0] = cluster * 10.0 + (i % 3) * 0.1;
            points[i][1] = cluster * -5.0 + (i % 2) * 0.1;
        }
        DataFrame X = DataFrame.ofNumeric(new String[]{"x", "y"}, points);

        KMeans kmeans = new KMeans(3, 300, 1e-6, 5, 42L);
        int[] labels = kmeans.fitPredict(X);

        assertEquals(3, kmeans.numberOfClusters());
        assertEquals(3, kmeans.clusterCenters().length);
        assertTrue(kmeans.inertia() >= 0.0);
        assertTrue(kmeans.iterationsRun() > 0);

        for (int i = 1; i < 30; i++) {
            assertEquals(labels[i / 10 * 10], labels[i]);
        }
        assertEquals(3, java.util.Arrays.stream(labels).distinct().count());

        int[] repeated = kmeans.fitPredict(X);
        assertArrayEquals(labels, repeated);
        assertArrayEquals(labels, kmeans.predict(X));
    }

    @Test
    void predictReturnsOneLabelPerRowAndValidatesInput() {
        DataFrame train = DataFrame.ofNumeric(new String[]{"x"}, new double[][]{
                {0}, {0.1}, {10}, {10.1}
        });
        KMeans kmeans = new KMeans(2, 100, 3, 1L);
        kmeans.fit(train);

        DataFrame test = DataFrame.ofNumeric(new String[]{"x"}, new double[][]{
                {0.05}, {10.05}, {5}
        });
        int[] labels = kmeans.predict(test);
        assertEquals(3, labels.length);
        assertNotEquals(labels[0], labels[1]);

        assertThrows(IllegalArgumentException.class,
                () -> new KMeans(5).fit(train));
        assertThrows(IllegalStateException.class, () -> new KMeans(2).predict(train));
    }
}
