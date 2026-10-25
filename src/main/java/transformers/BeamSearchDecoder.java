package transformers;

import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Beam search decoding for autoregressive generation.
 * Maintains top-k sequences at each step.
 */
public final class BeamSearchDecoder {

    private final GPTLMHeadModel model;
    private final int beamSize;
    private final int maxSeqLen;
    private final int vocabSize;

    public BeamSearchDecoder(GPTLMHeadModel model, int beamSize) {
        this.model = model;
        this.beamSize = beamSize;
        this.maxSeqLen = model.getGpt().maxSeqLen();
        this.vocabSize = model.vocabSize();
    }

    /**
     * Represents a candidate sequence during beam search.
     */
    public static final class Beam {
        public int[] tokens;
        public float logProb;
        public List<autograd.Tensor> pastK;
        public List<autograd.Tensor> pastV;
        public int length;

        public Beam(int[] tokens, float logProb, List<autograd.Tensor> pastK, List<autograd.Tensor> pastV) {
            this.tokens = tokens;
            this.logProb = logProb;
            this.pastK = pastK;
            this.pastV = pastV;
            this.length = tokens.length;
        }

        public Beam copy() {
            List<autograd.Tensor> newK = new ArrayList<>();
            List<autograd.Tensor> newV = new ArrayList<>();
            if (pastK != null) {
                for (Tensor k : pastK) newK.add(k != null ? k.detach() : null);
                for (Tensor v : pastV) newV.add(v != null ? v.detach() : null);
            }
            int[] newTokens = java.util.Arrays.copyOf(tokens, tokens.length);
            return new Beam(newTokens, logProb, newK, newV);
        }
    }

    /**
     * Performs beam search decoding.
     *
     * @param promptIds initial token ids
     * @param maxNewTokens maximum tokens to generate
     * @param lengthPenalty length penalty factor (alpha)
     * @return best generated sequence
     */
    public int[] generate(int[] promptIds, int maxNewTokens, float lengthPenalty) {
        if (promptIds.length > maxSeqLen) {
            throw new IllegalArgumentException("Prompt too long");
        }

        // Initialize beams with prompt
        List<Beam> beams = new ArrayList<>();
        Beam initial = new Beam(
            java.util.Arrays.copyOf(promptIds, promptIds.length),
            0.0f,
            new ArrayList<>(),
            new ArrayList<>()
        );
        for (int i = 0; i < model.getGpt().numLayers(); i++) {
            initial.pastK.add(null);
            initial.pastV.add(null);
        }
        beams.add(initial);

        int promptLen = promptIds.length;

        // Process prompt to build initial KV cache
        for (Beam beam : beams) {
            model.forward(beam.tokens, 1, beam.tokens.length,
                          beam.pastK, beam.pastV, 0);
        }

        // Generate
        int maxLen = promptIds.length + maxNewTokens;
        int maxSteps = maxNewTokens;

        for (int step = 0; step < maxSteps; step++) {
            List<Beam> candidates = new ArrayList<>();

            for (Beam beam : beams) {
                // Get next token logits
                int lastId = beam.tokens[beam.length - 1];
                int[] single = {lastId};
                Tensor logits = model.forward(
                    new int[]{lastId}, 1, 1,
                    beam.pastK, beam.pastV, beam.length - 1
                );

                float[] logitsData = logits.toFlat();

                // Get top beamSize next tokens
                int[] topIndices = topK(logitsData, beamSize);
                float[] logProbs = logSoftmax(logitsData);

                for (int idx : topIndices) {
                    int nextToken = idx;
                    float nextLogProb = logProbs[idx];

                    Beam newBeam = beam.copy();
                    int[] newTokens = java.util.Arrays.copyOf(newBeam.tokens, newBeam.length + 1);
                    newTokens[newBeam.length] = nextToken;
                    newBeam.tokens = newTokens;
                    newBeam.logProb += nextLogProb;
                    newBeam.length++;

                    candidates.add(newBeam);
                }
            }

            // Select top beamSize beams
            candidates.sort((a, b) -> Float.compare(
                b.logProb / (float) Math.pow(b.length, 0.6f),
                a.logProb / (float) Math.pow(a.length, 0.6f)
            ));

            beams = candidates.subList(0, Math.min(beamSize, candidates.size()));

            // Check if all beams finished (EOS token)
            boolean allDone = true;
            for (Beam beam : beams) {
                int lastToken = beam.tokens[beam.length - 1];
                if (lastToken != 50256) { // EOS token
                    allDone = false;
                    break;
                }
            }
            if (allDone) break;
        }

        // Return best beam
        Beam best = beams.stream()
            .max((a, b) -> Float.compare(
                a.logProb / (float) Math.pow(a.length, 0.6f),
                b.logProb / (float) Math.pow(b.length, 0.6f)
            ))
            .orElse(beams.get(0));

        return best.tokens;
    }

    private int[] topK(float[] arr, int k) {
        Integer[] indices = new Integer[arr.length];
        for (int i = 0; i < arr.length; i++) indices[i] = i;
        java.util.Arrays.sort(indices, (a, b) -> Float.compare(arr[b], arr[a]));
        int[] result = new int[Math.min(k, arr.length)];
        for (int i = 0; i < result.length; i++) result[i] = indices[i];
        return result;
    }

    private float[] logSoftmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        float[] out = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            out[i] = (float) Math.exp(logits[i] - max);
        }
        float sum = 0;
        for (float v : out) sum += v;
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) Math.log(out[i] / sum);
        }
        return out;
    }
}