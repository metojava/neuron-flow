package hyperparam;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Grid search and random search over a hyperparameter space.
 *
 * <p>Usage:
 * <pre>
 *   GridSearch gs = new GridSearch();
 *   gs.addParam("lr", new double[]{0.001, 0.01, 0.1});
 *   gs.addParam("hidden", new int[]{64, 128, 256});
 *   for (double[] params : gs.combinations()) {
 *       // train model with params[0], params[1], ...
 *   }
 * </pre>
 */
public final class GridSearch {

    private final List<String> names = new ArrayList<>();
    private final List<Object[]> values = new ArrayList<>();

    /** Adds a parameter with the given candidate values. */
    public void addParam(String name, Object[] candidates) {
        if (candidates == null || candidates.length == 0) {
            throw new IllegalArgumentException("candidates must not be empty");
        }
        names.add(name);
        values.add(candidates);
    }

    /** Returns the parameter names in order. */
    public String[] paramNames() {
        return names.toArray(new String[0]);
    }

    /** Returns all combinations as Object[][]. */
    public List<Object[]> combinations() {
        List<Object[]> result = new ArrayList<>();
        int[] indices = new int[names.size()];
        int total = 1;
        for (Object[] v : values) total *= v.length;

        for (int combo = 0; combo < total; combo++) {
            int rem = combo;
            Object[] params = new Object[names.size()];
            for (int i = names.size() - 1; i >= 0; i--) {
                int len = values.get(i).length;
                indices[i] = rem % len;
                rem /= len;
                params[i] = values.get(i)[indices[i]];
            }
            result.add(params);
        }
        return result;
    }

    /** Total number of combinations. */
    public int size() {
        int total = 1;
        for (Object[] v : values) total *= v.length;
        return total;
    }

    /**
     * Random search: samples {@code nSamples} random combinations
     * from the parameter space.
     */
    public static List<Object[]> randomSearch(GridSearch space, int nSamples,
                                               long seed) {
        Random rng = new Random(seed);
        List<Object[]> all = space.combinations();
        List<Object[]> result = new ArrayList<>();
        for (int i = 0; i < nSamples && i < all.size(); i++) {
            result.add(all.get(rng.nextInt(all.size())));
        }
        return result;
    }
}
