package ml;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Sequential preprocessing pipeline. Every stage is fitted on the output of
 * the previous stage, which is the standard way to prevent leakage such as
 * computing scaler statistics before imputation.
 */
public final class Pipeline implements Transformer {

    private final List<Transformer> transformers;
    private List<Transformer> fittedStages;
    private boolean fitted;

    public Pipeline(List<? extends Transformer> transformers) {
        Objects.requireNonNull(transformers, "transformers");
        if (transformers.isEmpty()) {
            throw new IllegalArgumentException("A pipeline needs at least one transformer");
        }
        List<Transformer> copy = new ArrayList<>(transformers.size());
        for (Transformer transformer : transformers) {
            copy.add(Objects.requireNonNull(transformer, "transformer"));
        }
        this.transformers = List.copyOf(copy);
    }

    public static Pipeline of(Transformer... transformers) {
        Objects.requireNonNull(transformers, "transformers");
        return new Pipeline(List.of(transformers));
    }

    /** Appends a stage and returns a new pipeline; this instance is unchanged. */
    public Pipeline then(Transformer transformer) {
        List<Transformer> stages = new ArrayList<>(transformers);
        stages.add(Objects.requireNonNull(transformer, "transformer"));
        return new Pipeline(stages);
    }

    public List<Transformer> transformers() {
        return transformers;
    }

    @Override
    public Pipeline fit(DataFrame X) {
        DataFrame current = X;
        List<Transformer> stages = new ArrayList<>(transformers.size());
        for (Transformer transformer : transformers) {
            Transformer fittedStage = transformer.fit(current);
            current = fittedStage.transform(current);
            stages.add(fittedStage);
        }
        fittedStages = List.copyOf(stages);
        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline has not been fitted");
        }
        DataFrame current = X;
        for (Transformer transformer : fittedStages) {
            current = transformer.transform(current);
        }
        return current;
    }

    @Override
    public boolean isFitted() {
        return fitted;
    }
}
