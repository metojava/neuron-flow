package layers;

import java.io.PrintStream;

/**
 * Prints a human-readable summary of a NeuralNetwork's architecture.
 */
public final class ModelSummary {

    private ModelSummary() {
    }

    /**
     * Prints layer sizes, activation functions, and parameter counts.
     *
     * @param out  output stream
     * @param net  the network to summarize
     */
    public static void print(PrintStream out, NeuralNetwork net) {
        int[] topo = net.getTopology();
        out.println("Model summary:");
        out.println("  Layer  Neurons  Params");
        int total = 0;
        for (int i = 0; i < topo.length; i++) {
            int params = (i == 0) ? 0 : topo[i] * topo[i - 1] + topo[i];
            total += params;
            out.printf("  %-5d  %-7d  %d%n", i, topo[i], params);
        }
        out.println("  Total parameters: " + total);
        out.println("  (Actual: " + net.getParameters().length + ")");
    }
}
