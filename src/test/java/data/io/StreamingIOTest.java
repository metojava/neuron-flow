package data.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class StreamingIOTest {

    @TempDir
    Path tmp;

    @Test
    void csvRoundTripInBatches() throws IOException {
        Path file = tmp.resolve("data.csv");
        float[][] rows = {{1f, 2.5f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}, {10f, 11f, 12f}, {13f, 14f, 15f}};
        try (CsvStreamWriter w = new CsvStreamWriter(file, ',', new String[] {"a", "b", "c"}, false)) {
            w.writeRows(rows);
            assertEquals(5, w.rowsWritten());
        }

        float[][] readBack = new float[0][];
        try (CsvStreamReader r = new CsvStreamReader(file)) {
            assertArrayEquals(new String[] {"a", "b", "c"}, r.header());
            float[][] batch;
            while ((batch = r.nextBatch(2)).length > 0) {
                float[][] merged = new float[readBack.length + batch.length][];
                System.arraycopy(readBack, 0, merged, 0, readBack.length);
                System.arraycopy(batch, 0, merged, readBack.length, batch.length);
                readBack = merged;
            }
            assertEquals(5, r.rowsRead());
        }
        assertEquals(rows.length, readBack.length);
        for (int i = 0; i < rows.length; i++) {
            assertArrayEquals(rows[i], readBack[i], 0f);
        }
    }

    @Test
    void csvQuotedFieldsAndNaN() throws IOException {
        Path file = tmp.resolve("quoted.csv");
        Files.writeString(file, "x,y,z\n\"1\",,3\n4,NA,\"6\"\n");
        try (CsvStreamReader r = new CsvStreamReader(file)) {
            float[][] b = r.nextBatch(10);
            assertEquals(2, b.length);
            assertEquals(1f, b[0][0]);
            assertTrue(Float.isNaN(b[0][1]));
            assertTrue(Float.isNaN(b[1][1]));
            assertEquals(6f, b[1][2]);
        }
    }

    @Test
    void csvMultilineQuotedField() throws IOException {
        Path file = tmp.resolve("multi.csv");
        Files.writeString(file, "a,b\n\"1\n\",2\n");
        try (CsvStreamReader r = new CsvStreamReader(file)) {
            float[][] b = r.nextBatch(5);
            assertEquals(1, b.length);
            assertEquals(1f, b[0][0]);
            assertEquals(2f, b[0][1]);
        }
    }

    @Test
    void csvRejectsNonNumeric() throws IOException {
        Path file = tmp.resolve("bad.csv");
        Files.writeString(file, "a\nhello\n");
        try (CsvStreamReader r = new CsvStreamReader(file)) {
            assertThrows(IOException.class, () -> r.nextBatch(1));
        }
    }

    @Test
    void jsonLinesRoundTrip() throws IOException {
        Path file = tmp.resolve("data.jsonl");
        float[][] rows = {{0.5f, 1f}, {Float.NaN, 2f}, {3f, 4f}};
        try (JsonLinesStream.Writer w = new JsonLinesStream.Writer(file)) {
            w.writeRows(rows);
        }
        String text = Files.readString(file);
        assertTrue(text.contains("null"));

        try (JsonLinesStream.Reader r = new JsonLinesStream.Reader(file)) {
            float[][] b = r.nextBatch(10);
            assertEquals(3, b.length);
            assertArrayEquals(rows[0], b[0], 0f);
            assertTrue(Float.isNaN(b[1][0]));
            assertArrayEquals(rows[2], b[2], 0f);
        }
    }

    @Test
    void jsonLinesRejectsObjects() throws IOException {
        Path file = tmp.resolve("obj.jsonl");
        Files.writeString(file, "{\"a\": 1}\n");
        try (JsonLinesStream.Reader r = new JsonLinesStream.Reader(file)) {
            assertThrows(IOException.class, () -> r.nextBatch(1));
        }
    }

    @Test
    void dropnaRemovesRowsWithNaN() {
        float[][] rows = {{1f, 2f}, {Float.NaN, 3f}, {4f, 5f}};
        float[][] out = MissingValues.dropna(rows);
        assertEquals(2, out.length);
        assertArrayEquals(new float[] {4f, 5f}, out[1], 0f);
    }

    @Test
    void fillnaReplacesNaN() {
        float[][] rows = {{1f, Float.NaN}, {Float.NaN, 3f}};
        float[][] out = MissingValues.fillna(rows, -1f);
        assertArrayEquals(new float[] {1f, -1f}, out[0], 0f);
        assertArrayEquals(new float[] {-1f, 3f}, out[1], 0f);
    }

    @Test
    void fillnaMeanUsesColumnMeans() {
        float[][] rows = {{1f, Float.NaN}, {3f, 4f}, {Float.NaN, 8f}};
        float[][] out = MissingValues.fillnaMean(rows);
        assertArrayEquals(new float[] {1f, 6f}, out[0], 0f);
        assertArrayEquals(new float[] {3f, 4f}, out[1], 0f);
        assertArrayEquals(new float[] {2f, 8f}, out[2], 0f); // col 0 mean of 1,3
    }
}
