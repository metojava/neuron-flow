package ml;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DataFrameTest {

    @Test
    void buildsMixedFrameAndPreservesColumnOrder() {
        DataFrame frame = DataFrame.builder(2)
                .addNumeric("age", new double[]{30, 40})
                .addCategorical("city", new String[]{"Paris", "Berlin"})
                .build();

        assertEquals(2, frame.rows());
        assertEquals(2, frame.columns());
        assertArrayEquals(new String[]{"age", "city"}, frame.names());
        assertArrayEquals(new ColumnType[]{ColumnType.NUMERIC, ColumnType.CATEGORICAL}, frame.types());
        assertEquals(30.0, frame.getDouble(0, 0));
        assertEquals("Berlin", frame.getString(1, 1));
        assertEquals(1, frame.columnIndex("city"));
        assertFalse(frame.allNumeric());
        assertThrows(IllegalStateException.class, frame::toDoubleMatrix);
    }

    @Test
    void accessorsReturnDefensiveCopies() {
        DataFrame frame = DataFrame.ofNumeric(new String[]{"x"}, new double[][]{{1.0}, {2.0}});
        double[] column = frame.numericColumn(0);
        column[0] = 99.0;
        assertEquals(1.0, frame.getDouble(0, 0));

        int[] rows = {1};
        DataFrame subset = frame.takeRows(rows);
        rows[0] = 0;
        assertEquals(2.0, subset.getDouble(0, 0));

        double[][] matrix = frame.toDoubleMatrix();
        matrix[0][0] = 42.0;
        assertEquals(1.0, frame.getDouble(0, 0));
    }

    @Test
    void selectingColumnsKeepsTypesAndRejectsDuplicates() {
        DataFrame frame = DataFrame.builder(2)
                .addNumeric("a", new double[]{1, 2})
                .addCategorical("b", new String[]{"x", "y"})
                .addNumeric("c", new double[]{3, 4})
                .build();
        DataFrame selected = frame.select(2, 1);
        assertArrayEquals(new String[]{"c", "b"}, selected.names());
        assertEquals(ColumnType.CATEGORICAL, selected.type(1));
        assertThrows(IllegalArgumentException.class, () -> frame.select(0, 0));
    }

    @Test
    void validatesShapeAndNames() {
        assertThrows(IllegalArgumentException.class,
                () -> DataFrame.ofNumeric(new String[]{"a", "b"}, new double[][]{{1.0}}));
        assertThrows(IllegalArgumentException.class,
                () -> DataFrame.ofNumeric(new String[]{"a", "a"}, new double[][]{{1.0}}));
        assertThrows(IllegalArgumentException.class,
                () -> DataFrame.ofNumeric(new String[]{"a"}, new double[][]{{1.0}, {2.0, 3.0}}));
        assertThrows(IllegalStateException.class,
                () -> DataFrame.builder(1).build());
        DataFrame frame = DataFrame.ofNumeric(new String[]{"a"}, new double[][]{{1.0}});
        assertThrows(IndexOutOfBoundsException.class, () -> frame.getDouble(0, 5));
    }
}
