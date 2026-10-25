package util;

import java.util.Random;

/**
 * Centralized seed management for reproducibility. Stores a global
 * seed and provides seeded RNGs for all components.
 */
public final class SeedManager {

    private static long globalSeed = 42L;

    private SeedManager() {
    }

    /** Sets the global seed. */
    public static void setSeed(long seed) {
        globalSeed = seed;
    }

    /** Returns the global seed. */
    public static long getSeed() {
        return globalSeed;
    }

    /** Returns a new Random seeded with the global seed + offset. */
    public static Random rng(long offset) {
        return new Random(globalSeed + offset);
    }

    /** Returns a new Random seeded with the global seed. */
    public static Random rng() {
        return new Random(globalSeed);
    }

    /** Combines multiple seeds into one. */
    public static long combine(long... seeds) {
        long result = 0L;
        for (long s : seeds) {
            result = result * 31 + s;
        }
        return result;
    }
}
