import data.io.parquet.ParquetFileReader;
import java.nio.file.Path;

/**
 * Inspect the yellow taxi parquet file structure.
 */
public class InspectTaxiData {
    public static void main(String[] args) throws Exception {
        Path file = Path.of("projects", "data", "yellow_tripdata_2025-01.parquet");
        try (data.io.parquet.ParquetFileReader reader = new data.io.parquet.ParquetFileReader(Path.of("projects", "data", "yellow_tripdata_2025-01.parquet"))) {
            System.out.println("Columns: " + java.util.Arrays.toString(reader.header()));
            System.out.println("Rows: " + reader.numRows());
            System.out.println("Row groups: " + reader.rowGroupCount());
            float[][] batch = reader.nextBatch(5);
            for (float[] row : batch) {
                System.out.println(java.util.Arrays.toString(row));
            }
        }
    }
}