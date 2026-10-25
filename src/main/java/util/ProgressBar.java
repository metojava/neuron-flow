package util;

/**
 * Simple console progress bar. Prints a bar that updates in place
 * using carriage return.
 */
public final class ProgressBar {

    private final int total;
    private final int width;
    private int current;

    /**
     * @param total total number of steps
     * @param width bar width in characters
     */
    public ProgressBar(int total, int width) {
        if (total <= 0) throw new IllegalArgumentException("total must be positive");
        this.total = total;
        this.width = width;
        this.current = 0;
    }

    /** Default width of 40 characters. */
    public ProgressBar(int total) {
        this(total, 40);
    }

    /**
     * Advances the bar by one step and prints it.
     *
     * @param label optional label to display after the bar
     */
    public void step(String label) {
        current++;
        if (current > total) current = total;
        int filled = (int) ((double) current / total * width);
        StringBuilder bar = new StringBuilder("\r[");
        for (int i = 0; i < width; i++) {
            bar.append(i < filled ? '=' : ' ');
        }
        bar.append("] ").append(current).append("/").append(total);
        if (label != null && !label.isEmpty()) {
            bar.append("  ").append(label);
        }
        System.out.print(bar);
        if (current >= total) {
            System.out.println();
        }
    }

    /** Advances without a label. */
    public void step() {
        step("");
    }

    /** Resets the bar to zero. */
    public void reset() {
        current = 0;
    }

    /** Returns the current progress (0..total). */
    public int current() {
        return current;
    }
}
