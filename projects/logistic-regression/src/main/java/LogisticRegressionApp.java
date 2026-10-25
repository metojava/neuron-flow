import data.io.parquet.ParquetFileReader;
import data.io.parquet.ParquetFileWriter;
import ml.DataFrame;
import ml.Vector;
import ml.model.LogisticRegression;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;

/**
 * Logistic Regression demo with Parquet file I/O.
 */
public final class LogisticRegressionApp {

    private static final String[] COLUMNS = {"feature1", "feature2", "feature3", "feature4", "label"};

    public static void main(String[] args) throws IOException {
        Path dataDir = Paths.get("data", "parquet");
        java.nio.file.Files.createDirectories(dataDir);
        Path file = Paths.get("data", "parquet", "logistic_regression_demo.parquet");

        int trainRows = 2000;
        int testRows = 500;

        System.out.println("=== Logistic Regression with Parquet I/O Demo ===\n");

        // Step 1: Write synthetic dataset to Parquet
        System.out.println("Step 1: Writing synthetic dataset to Parquet...");
        writeDataset(Paths.get("data", "parquet", "logistic_regression_demo.parquet"), trainRows + testRows);

        // Step 2: Read and split into train/test
        System.out.println("\nStep 2: Reading dataset and splitting train/test...");
        float[][] allData = readAllData(file, trainRows + testRows);
        
        // Shuffle and split
        shuffleRows(allData, 42L);
        float[][] trainData = Arrays.copyOfRange(allData, 0, trainRows);
        float[][] testData = Arrays.copyOfRange(allData, trainRows, trainRows + testRows);

        System.out.printf("Train rows: %d, Test rows: %d%n", trainData.length, testData.length);

        // Step 3: Convert to DataFrame/Vector and train logistic regression
        System.out.println("\nStep 3: Training logistic regression model...");
        DataFrame trainDF = toDataFrame(trainData);
        Vector trainLabels = toVector(extractLabels(trainData));
        
        LogisticRegression model = new LogisticRegression(0.1, 1000, 0.0, 1e-6, true);
        model.fit(trainDF, trainLabels);

        // Step 4: Evaluate on test set
        System.out.println("\nStep 4: Evaluating on test set...");
        DataFrame testDF = toDataFrame(testData);
        Vector testLabels = toVector(extractLabels(testData));
        
        double[] predictions = model.predict(testDF);
        double[][] probs = model.predictProbabilities(testDF);
        float[] trueLabels = extractLabels(testData);
        
        evaluate(predictions, trueLabels);

        // Step 5: Demonstrate Parquet batch reading
        System.out.println("\nStep 5: Demonstrating Parquet batch reading...");
        demonstrateBatchReading();
    }

    private static void writeDataset(Path file, int rows) throws IOException {
        Random random = new Random(42L);

        try (ParquetFileWriter writer = new ParquetFileWriter(file, COLUMNS, COLUMNS.length, 256)) {
            for (int i = 0; i < rows; i++) {
                float f1 = (float) random.nextGaussian();
                float f2 = (float) random.nextGaussian();
                float f3 = (float) random.nextGaussian();
                float f4 = (float) random.nextGaussian();
                
                float label = (f1 + f2 > f3 + f4) ? 1.0f : 0.0f;
                
                if (random.nextDouble() < 0.1) {
                    label = 1.0f - label;
                }

                writer.writeRow(new float[]{f1, f2, f3, f4, label});
            }

            System.out.println("  Written to: " + file.toAbsolutePath());
            System.out.println("  Columns: " + java.util.Arrays.toString(writer.header()));
            System.out.println("  Rows written: " + writer.rowsWritten());
        }
    }

    private static float[][] readAllData(Path file, int expectedRows) throws IOException {
        try (ParquetFileReader reader = new ParquetFileReader(Paths.get(file.toString()))) {
            System.out.println("  Reading from: " + Paths.get(file.toString()).toAbsolutePath());
            System.out.println("  Columns: " + java.util.Arrays.toString(reader.header()));
            System.out.println("  Total rows in file: " + reader.numRows());

            float[][] allData = new float[expectedRows][5];
            int rowsRead = 0;

            float[][] batch;
            while ((batch = reader.nextBatch(1024)).length > 0) {
                for (float[] row : batch) {
                    if (rowsRead < expectedRows) {
                        allData[rowsRead] = row;
                        rowsRead++;
                    }
                }
            }

            if (rowsRead != expectedRows) {
                throw new IllegalStateException("Expected " + expectedRows + " rows but read " + rowsRead);
            }
            System.out.println("  Successfully read " + rowsRead + " rows");
            return allData;
        }
    }

    private static void shuffleRows(float[][] data, long seed) {
        Random random = new Random(seed);
        for (int i = data.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            float[] temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }

    private static float[] extractLabels(float[][] data) {
        float[] labels = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            labels[i] = data[i][4];
        }
        return labels;
    }

    private static DataFrame toDataFrame(float[][] data) {
        int rows = data.length;
        double[][] matrix = new double[rows][4];
        String[] names = {"feature1", "feature2", "feature3", "feature4"};
        
        for (int i = 0; i < rows; i++) {
            matrix[i][0] = data[i][0];
            matrix[i][1] = data[i][1];
            matrix[i][2] = data[i][2];
            matrix[i][3] = data[i][3];
        }
        return DataFrame.ofNumeric(names, matrix);
    }

    private static Vector toVector(float[] labels) {
        double[] dLabels = new double[labels.length];
        for (int i = 0; i < labels.length; i++) {
            dLabels[i] = labels[i];
        }
        return Vector.of(dLabels);
    }

    private static void evaluate(double[] predictions, float[] trueLabels) {
        int correct = 0;
        int tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (int i = 0; i < predictions.length; i++) {
            int pred = (int) predictions[i];
            int actual = (int) trueLabels[i];
            
            if (pred == actual) correct++;
            
            if (pred == 1 && actual == 1) tp++;
            else if (pred == 1 && actual == 0) fp++;
            else if (pred == 0 && actual == 0) tn++;
            else if (pred == 0 && actual == 1) fn++;
        }

        float accuracy = (float) correct / predictions.length;
        float precision = tp / (float) Math.max(1, tp + fp);
        float recall = tp / (float) Math.max(1, tp + fn);
        float f1 = 2 * precision * recall / Math.max(1e-6f, precision + recall);

        System.out.printf("  Accuracy:  %.4f%n", accuracy);
        System.out.printf("  Precision: %.4f%n", precision);
        System.out.printf("  Recall:    %.4f%n", recall);
        System.out.printf("  F1 Score:  %.4f%n", f1);
        System.out.printf("  Confusion Matrix: TP=%d, FP=%d, TN=%d, FN=%d%n", tp, fp, tn, fn);
    }

    private static void demonstrateBatchReading() throws IOException {
        Path file = Paths.get("data", "parquet", "logistic_regression_demo.parquet");
        
        try (ParquetFileReader reader = new ParquetFileReader(Paths.get("data", "parquet", "logistic_regression_demo.parquet"))) {
            System.out.println("\n  Reading in batches of 500:");
            int batchNum = 0;
            long totalRows = 0;
            
            while (true) {
                float[][] batch = reader.nextBatch(500);
                if (batch.length == 0) break;
                batchNum++;
                totalRows += batch.length;
                if (batchNum <= 3) {
                    System.out.printf("  Batch %d: %d rows (first row: %s)%n", 
                        batchNum, batch.length, Arrays.toString(batch[0]));
                }
            }
            System.out.printf("  Total batches: %d, Total rows read: %d%n", batchNum, totalRows);
        }
    }
}