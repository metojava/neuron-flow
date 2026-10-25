package layers;

/**
 * Supported serialization formats for saving and loading
 * trained models. Each format has different trade-offs:
 * <ul>
 *   <li>{@link #BINARY} — compact, fast, little-endian floats.
 *       Best for production deployment.</li>
 *   <li>{@link #JSON} — human-readable, diffable, easy to
 *       inspect or load from other languages. Larger files.</li>
 *   <li>{@link #CSV} — one parameter per line. Opens in any
 *       spreadsheet. Useful for debugging weight distributions.</li>
 * </ul>
 */
public enum ModelFormat {

    /** Compact binary: magic int, layer sizes, raw floats. */
    BINARY(".bin"),

    /** Human-readable JSON with topology and parameters. */
    JSON(".json"),

    /** One weight or bias value per CSV line. */
    CSV(".csv");

    private final String extension;

    ModelFormat(String extension) {
        this.extension = extension;
    }

    /** @return the canonical file extension, e.g. ".bin" */
    public String extension() {
        return extension;
    }

    /**
     * Infers the format from a file path's extension.
     *
     * @param path file path
     * @return the matching format, or {@link #BINARY} as default
     */
    public static ModelFormat fromPath(String path) {
        if (path == null) {
            return BINARY;
        }

        String lower = path.toLowerCase();

        if (lower.endsWith(".json")) {
            return JSON;
        }

        if (lower.endsWith(".csv")) {
            return CSV;
        }

        return BINARY;
    }
}
