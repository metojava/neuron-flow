package ml;

import ml.preprocessing.MinMaxScaler;
import ml.preprocessing.OneHotEncoder;
import ml.preprocessing.SimpleImputer;
import ml.preprocessing.StandardScaler;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PreprocessingTest {

    @Test
    void standardScalerUsesTrainingStatisticsAndHandlesConstantColumns() {
        DataFrame train = DataFrame.ofNumeric(new String[]{"x", "constant"},
                new double[][]{{0, 5}, {2, 5}});
        StandardScaler scaler = new StandardScaler();
        DataFrame scaled = scaler.fitTransform(train);

        assertArrayEquals(new double[]{-1.0, 1.0}, scaled.numericColumn(0), 1e-9);
        assertArrayEquals(new double[]{0.0, 0.0}, scaled.numericColumn(1), 1e-9);

        DataFrame unseen = DataFrame.ofNumeric(new String[]{"x", "constant"},
                new double[][]{{10, 5}});
        double[] values = scaler.transform(unseen).numericColumn(0);
        assertEquals(9.0, values[0], 1e-9);
        assertEquals(0.0, scaler.transform(unseen).getDouble(0, 1), 1e-9);
        assertThrows(IllegalArgumentException.class,
                () -> scaler.transform(DataFrame.ofNumeric(new String[]{"x"}, new double[][]{{1}})));
    }

    @Test
    void minMaxScalerSupportsCustomRangeAndConstantColumns() {
        DataFrame train = DataFrame.ofNumeric(new String[]{"a", "b"},
                new double[][]{{0, 3}, {10, 3}});
        MinMaxScaler scaler = new MinMaxScaler(-1, 1);
        DataFrame scaled = scaler.fitTransform(train);
        assertArrayEquals(new double[]{-1, 1}, scaled.numericColumn(0), 1e-9);
        assertArrayEquals(new double[]{-1, -1}, scaled.numericColumn(1), 1e-9);
        assertThrows(IllegalArgumentException.class, () -> new MinMaxScaler(1, 1));
    }

    @Test
    void imputerSupportsMeanMedianAndMostFrequent() {
        DataFrame numeric = DataFrame.ofNumeric(new String[]{"x"},
                new double[][]{{1}, {Double.NaN}, {3}});
        SimpleImputer mean = new SimpleImputer(SimpleImputer.Strategy.MEAN);
        assertEquals(2.0, mean.fitTransform(numeric).getDouble(1, 0), 1e-9);
        SimpleImputer median = new SimpleImputer(SimpleImputer.Strategy.MEDIAN);
        assertEquals(2.0, median.fitTransform(numeric).getDouble(1, 0), 1e-9);

        DataFrame categorical = DataFrame.ofCategorical(new String[]{"city"},
                new String[][]{{"a"}, {null}, {"a"}, {"b"}});
        SimpleImputer frequent = new SimpleImputer(SimpleImputer.Strategy.MOST_FREQUENT);
        assertEquals("a", frequent.fitTransform(categorical).getString(1, 0));
        assertEquals("a", new SimpleImputer(SimpleImputer.Strategy.MEAN)
                .fitTransform(categorical).getString(1, 0));
    }

    @Test
    void oneHotEncoderDropsFirstLevelAndHandlesUnknowns() {
        DataFrame train = DataFrame.ofCategorical(new String[]{"color"},
                new String[][]{{"red"}, {"blue"}, {"green"}, {"blue"}});
        OneHotEncoder encoder = new OneHotEncoder();
        DataFrame encoded = encoder.fitTransform(train);
        assertArrayEquals(new String[]{"color[green]", "color[red]"}, encoded.names());
        assertArrayEquals(new double[]{0, 1}, encoded.toDoubleMatrix()[0], 1e-9);
        assertArrayEquals(new double[]{0, 0}, encoded.toDoubleMatrix()[1], 1e-9);
        assertArrayEquals(new double[]{1, 0}, encoded.toDoubleMatrix()[2], 1e-9);

        DataFrame unseen = DataFrame.ofCategorical(new String[]{"color"},
                new String[][]{{"purple"}});
        assertArrayEquals(new double[]{0, 0}, encoder.transform(unseen).toDoubleMatrix()[0], 1e-9);

        OneHotEncoder strict = new OneHotEncoder(true, OneHotEncoder.UnknownStrategy.ERROR);
        strict.fit(train);
        assertThrows(IllegalArgumentException.class, () -> strict.transform(unseen));
    }

    @Test
    void pipelineFitsSequentiallyAndRejectsUseBeforeFit() {
        DataFrame train = DataFrame.builder(3)
                .addNumeric("age", new double[]{20, 30, Double.NaN})
                .addCategorical("city", new String[]{"a", "b", "a"})
                .build();
        Pipeline pipeline = Pipeline.of(
                new SimpleImputer(SimpleImputer.Strategy.MOST_FREQUENT),
                new OneHotEncoder(),
                new StandardScaler());
        DataFrame transformed = pipeline.fitTransform(train);
        assertTrue(pipeline.isFitted());
        assertEquals(3, transformed.rows());
        assertEquals(2, transformed.columns());
        assertTrue(transformed.allNumeric());
        assertThrows(IllegalStateException.class, () -> Pipeline.of(new StandardScaler())
                .transform(train));
    }
}
