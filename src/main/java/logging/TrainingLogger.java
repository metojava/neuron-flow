package logging;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Logs training metrics to a CSV file. Each call to log() appends a row.
 *
 * <p>Usage:
 * <pre>
 *   TrainingLogger logger = new TrainingLogger(Path.of("metrics.csv"),
 *       "epoch", "train_loss", "val_acc", "lr");
 *   logger.log(1, 0.5, 0.85, 0.001);
 * </pre>
 */
public final class TrainingLogger {

    private final Path path;
    private final String[] columns;
    private boolean headerWritten;

    /**
     * @param path    CSV file path
     * @param columns column names
     */
    public TrainingLogger(Path path, String... columns) {
        if (path == null || columns == null || columns.length == 0) {
            throw new IllegalArgumentException("path and columns must not be empty");
        }
        this.path = path;
        this.columns = columns.clone();
        this.headerWritten = Files.exists(path);
    }

    /**
     * Appends a row of metrics. Values are converted to strings.
     *
     * @param values one value per column
     */
    public synchronized void log(Object... values) {
        if (values.length != columns.length) {
            throw new IllegalArgumentException(
                "Expected " + columns.length + " values, got " + values.length);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (!headerWritten) {
                writer.write(String.join(",", columns));
                writer.newLine();
                headerWritten = true;
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < values.length; i++) {
                if (i > 0) sb.append(",");
                sb.append(values[i]);
            }
            writer.write(sb.toString());
            writer.newLine();
        } catch (IOException e) {
            throw new RuntimeException("Failed to write log", e);
        }
    }

    /** Returns the log file path. */
    public Path path() {
        return path;
    }
}
