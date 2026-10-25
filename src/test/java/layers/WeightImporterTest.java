package layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import data.model.Tensor;
import org.junit.jupiter.api.Test;

class WeightImporterTest {

    @Test
    void denseTransposesTorchLayout() {
        // PyTorch Linear(3 -> 2): weight [out=2, in=3], bias [2]
        Tensor w = new Tensor("fc.weight", new int[] {2, 3}, new float[] {1, 2, 3, 4, 5, 6});
        Tensor b = new Tensor("fc.bias", new int[] {2}, new float[] {0.5f, -0.5f});
        Dense d = new Dense(3, 2, 1L);
        WeightImporter.loadDense(d, w, b);

        assertEquals(2f, d.weight(1, 0), 0f);  // input 1 -> output 0 = W[0][1]
        assertEquals(6f, d.weight(2, 1), 0f);  // input 2 -> output 1 = W[1][2]

        float[] y = d.forward(new float[] {1f, 0f, -1f});
        // W x + b: [1 - 3 + 0.5, 4 - 6 - 0.5]
        assertArrayEquals(new float[] {-1.5f, -2.5f}, y, 1e-6f);
    }

    @Test
    void denseNullBiasIsZero() {
        Tensor w = new Tensor("w", new int[] {1, 2}, new float[] {3f, 4f});
        Dense d = new Dense(2, 1, 1L);
        WeightImporter.loadDense(d, w, null);
        assertArrayEquals(new float[] {3f * 1 + 4f * 2}, d.forward(new float[] {1f, 2f}), 1e-6f);
    }

    @Test
    void denseRejectsWrongShapeAndLeavesLayerUnchanged() {
        Dense d = new Dense(3, 2, 1L);
        float[] before = new float[d.parameterCount()];
        d.copyParametersTo(before, 0);

        Tensor bad = new Tensor("bad", new int[] {3, 2}, new float[6]);
        assertThrows(IllegalArgumentException.class, () -> WeightImporter.loadDense(d, bad, null));

        float[] after = new float[d.parameterCount()];
        d.copyParametersTo(after, 0);
        assertArrayEquals(before, after, 0f);
    }

    @Test
    void conv2dCopiesKernelsAndBiases() {
        Conv2D c = new Conv2D(1, 1, 2, 2, 1, 0, 1L);
        Tensor w = new Tensor("conv.weight", new int[] {1, 1, 2, 2}, new float[] {1, 1, 1, 1});
        Tensor b = new Tensor("conv.bias", new int[] {1}, new float[] {0.5f});
        WeightImporter.loadConv2D(c, w, b);

        float[][][] out = c.forward(new float[][][] {{{1, 1}, {1, 1}}});
        assertEquals(4.5f, out[0][0][0], 1e-6f);
    }

    @Test
    void embeddingLoadsTable() {
        Embedding e = new Embedding(3, 2, 1L);
        Tensor table = new Tensor("emb", new int[] {3, 2}, new float[] {0, 1, 2, 3, 4, 5});
        WeightImporter.loadEmbedding(e, table);

        float[][] out = e.forward(new int[] {2, 0});
        assertArrayEquals(new float[] {4f, 5f}, out[0], 0f);
        assertArrayEquals(new float[] {0f, 1f}, out[1], 0f);
    }

    @Test
    void embeddingRejectsWrongShape() {
        Embedding e = new Embedding(3, 2, 1L);
        Tensor table = new Tensor("emb", new int[] {2, 3}, new float[6]);
        assertThrows(IllegalArgumentException.class, () -> WeightImporter.loadEmbedding(e, table));
    }

    @Test
    void layerNormLoadsGammaAndBeta() {
        LayerNorm ln = new LayerNorm(2);
        Tensor g = new Tensor("ln.weight", new int[] {2}, new float[] {2f, 2f});
        Tensor bt = new Tensor("ln.bias", new int[] {2}, new float[] {1f, 1f});
        WeightImporter.loadLayerNorm(ln, g, bt, 2);

        // input [1, 3]: mean 2, std ~1 -> normalized [-1, 1]; gamma 2, beta 1 -> [-1, 3]
        float[] y = ln.forward(new float[] {1f, 3f});
        assertArrayEquals(new float[] {-1f, 3f}, y, 1e-3f);
    }
}
