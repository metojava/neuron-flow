package data.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ColumnarIOTest {

    @TempDir
    Path tmp;

    private static final float[][] ROWS = {
            {1f, 2.5f, 3f},
            {4f, Float.NaN, 6f},
            {7f, 8f, -9.25f},
    };

    @Test
    void parquetRoundTripWithNulls() throws IOException {
        Path file = tmp.resolve("data.parquet");
        try (ParquetIO.Writer w = new ParquetIO.Writer(file, new String[] {"a", "b", "c"}, 3)) {
            w.writeRows(ROWS);
            assertEquals(3, w.rowsWritten());
        }
        try (ParquetIO.Reader r = new ParquetIO.Reader(file)) {
            assertArrayEquals(new String[] {"a", "b", "c"}, r.header());
            float[][] got = r.nextBatch(10);
            assertEquals(3, got.length);
            for (int i = 0; i < ROWS.length; i++) {
                assertArrayEquals(ROWS[i], got[i], 0f);
            }
            assertEquals(0, r.nextBatch(10).length);
        }
    }

    @Test
    void parquetReadsInBatches() throws IOException {
        Path file = tmp.resolve("batches.parquet");
        try (ParquetIO.Writer w = new ParquetIO.Writer(file, null, 2)) {
            for (int i = 0; i < 25; i++) {
                w.writeRow(new float[] {i, i * 2f});
            }
        }
        int total = 0;
        try (ParquetIO.Reader r = new ParquetIO.Reader(file)) {
            float[][] b;
            while ((b = r.nextBatch(10)).length > 0) {
                assertTrue(b.length <= 10);
                total += b.length;
            }
        }
        assertEquals(25, total);
    }

    @Test
    void parquetHandlesMultipleRowGroups() throws IOException {
        Path file = tmp.resolve("rowgroups.parquet");
        try (ParquetIO.Writer w = new ParquetIO.Writer(file, null, 2, 3)) {
            for (int i = 0; i < 10; i++) {
                w.writeRow(new float[] {i, i * 2f});
            }
        }
        int total = 0;
        try (ParquetIO.Reader r = new ParquetIO.Reader(file)) {
            assertEquals(10, r.numRows());
            float[][] b;
            while ((b = r.nextBatch(4)).length > 0) {
                assertTrue(b.length <= 4);
                for (float[] row : b) {
                    assertEquals(total, row[0]);
                    assertEquals(total * 2f, row[1]);
                    total++;
                }
            }
        }
        assertEquals(10, total);
    }

    @Test
    void parquetHandlesEmptyFile() throws IOException {
        Path file = tmp.resolve("empty.parquet");
        try (ParquetIO.Writer w = new ParquetIO.Writer(file, new String[] {"a"}, 1)) {
            assertEquals(0, w.rowsWritten());
        }
        try (ParquetIO.Reader r = new ParquetIO.Reader(file)) {
            assertArrayEquals(new String[] {"a"}, r.header());
            assertEquals(0, r.numRows());
            assertEquals(0, r.nextBatch(10).length);
        }
    }
}
