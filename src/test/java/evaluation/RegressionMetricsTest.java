package evaluation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RegressionMetricsTest {
    @Test
    void computesCommonRegressionScores() {
        float[] predicted = {1, 3, 5};
        float[] actual = {2, 2, 4};
        assertEquals(1.0, RegressionMetrics.meanAbsoluteError(predicted, actual), 1e-12);
        assertEquals(1.0, RegressionMetrics.meanSquaredError(predicted, actual), 1e-12);
        assertEquals(1.0, RegressionMetrics.rootMeanSquaredError(predicted, actual), 1e-12);
        assertEquals(-0.125, RegressionMetrics.rSquared(predicted, actual), 1e-12);
    }

    @Test
    void handlesConstantTargetsAndRejectsInvalidValues() {
        assertEquals(1.0, RegressionMetrics.rSquared(new float[] {2, 2}, new float[] {2, 2}));
        assertEquals(0.0, RegressionMetrics.rSquared(new float[] {2, 3}, new float[] {2, 2}));
        assertThrows(IllegalArgumentException.class,
            () -> RegressionMetrics.meanSquaredError(new float[] {Float.NaN}, new float[] {1}));
    }
}
