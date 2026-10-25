import data.io.parquet.ParquetFileReader;
import data.io.parquet.ParquetFileWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

/**
 * Project 12 - Parquet file I/O.
 *
 * Writes a small numeric dataset to data/parquet/people.parquet, reads it
 * back in batches, and prints a summary. Every column is stored as nullable
 * FLOAT; Float.NaN is written as a Parquet null and read back as NaN.
 *
 * Run from this folder with: mvn install -DskipTests in the repository root,
 * then: mvn -f projects/parquet/pom.xml compile exec:exec
 *
 * The older convenience wrapper data.io.ParquetIO exposes the same reader and
 * writer with a simpler package-level name.
 */
public class ParquetApp {

    private static final String[] COLUMNS = {"height_cm", "weight_kg", "age", "label"};

    public static void main(String[] args) throws IOException {
        Path file = Path.of("data", "parquet", "people.parquet");
        int rows = 2_000;

        writeDataset(file, rows);
        readDataset(file, rows);
    }

    private static void writeDataset(Path file, int rows) throws IOException {
        Random random = new Random(42L);

        try (ParquetFileWriter writer = new ParquetFileWriter(file, COLUMNS, COLUMNS.length, 256)) {
            for (int i = 0; i < rows; i++) {
                float height = 150f + random.nextFloat() * 50f;
                float weight = 40f + random.nextFloat() * 80f;
                float age = 18f + random.nextInt(60);
                float label = weight / ((height / 100f) * (height / 100f));

                // Every 53rd weight is missing; the writer stores it as null.
                if (i % 53 == 0) {
                    weight = Float.NaN;
                }
                writer.writeRow(new float[] {height, weight, age, label});
            }

            System.out.println("Writing " + file.toAbsolutePath());
            System.out.println("  columns:    " + Arrays.toString(writer.header()));
            System.out.println("  rows:       " + writer.rowsWritten());
        }
    }

    private static void readDataset(Path file, int expectedRows) throws IOException {
        long rowsRead = 0;
        long nullWeights = 0;
        double weightSum = 0;
        long weightCount = 0;
        int sampleRows = 0;

        try (ParquetFileReader reader = new ParquetFileReader(file)) {
            System.out.println();
            System.out.println("Reading " + file.toAbsolutePath());
            System.out.println("  columns:    " + Arrays.toString(reader.header()));
            System.out.println("  rows:       " + reader.numRows());
            System.out.println("  row groups: " + reader.rowGroupCount());

            float[][] batch;
            while ((batch = reader.nextBatch(128)).length > 0) {
                for (float[] row : batch) {
                    if (sampleRows < 5) {
                        System.out.printf("  %s%n", Arrays.toString(row));
                        sampleRows++;
                    }
                    if (Float.isNaN(row[1])) {
                        nullWeights++;
                    } else {
                        weightSum += row[1];
                        weightCount++;
                    }
                    rowsRead++;
                }
            }
        }

        if (rowsRead != expectedRows) {
            throw new IllegalStateException("Expected " + expectedRows + " rows but read " + rowsRead);
        }
        System.out.printf("  read rows:  %d%n", rowsRead);
        System.out.printf("  null weights: %d%n", nullWeights);
        System.out.printf("  mean weight:  %.2f%n", weightSum / weightCount);
    }
}
