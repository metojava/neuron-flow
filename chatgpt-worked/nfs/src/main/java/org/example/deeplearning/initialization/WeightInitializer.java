package org.example.deeplearning.initialization;

import java.util.Random;

/** Produces one incoming weight for a dense layer. */
@FunctionalInterface
public interface WeightInitializer {
    double sample(int fanIn, int fanOut, Random random);
    static WeightInitializer xavierUniform() {
        return (fanIn, fanOut, random) -> {
            double limit = Math.sqrt(6.0 / (fanIn + fanOut));
            return (random.nextDouble() * 2.0 - 1.0) * limit;
        };
    }
    static WeightInitializer heNormal() {
        return (fanIn, fanOut, random) -> random.nextGaussian() * Math.sqrt(2.0 / fanIn);
    }
}
