package layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import data.model.SafetensorsIO;
import data.model.Tensor;
import data.model.WeightFile;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import optim.Adam;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelLoaderTest {

    @TempDir
    Path tmp;

    private static float[][] random(int rows, int cols, long seed) {
        Random r = new Random(seed);
        float[][] m = new float[rows][cols];
        for (float[] row : m) {
            for (int j = 0; j < cols; j++) row[j] = (float) r.nextGaussian();
        }
        return m;
    }

    private static double attentionLoss(SelfAttention a, float[][] x, float[][] g) {
        float[][] out = a.forward(x);
        double s = 0;
        for (int t = 0; t < out.length; t++) {
            for (int d = 0; d < out[t].length; d++) s += (double) g[t][d] * out[t][d];
        }
        return s;
    }

    @Test
    void selfAttentionGradientsMatchFiniteDifferences() {
        int d = 3;
        SelfAttention a = new SelfAttention(d, 5L);
        float[][] x = random(4, d, 6L);
        float[][] g = random(4, d, 7L);
        float eps = 1e-2f;

        a.zeroGrad();
        a.forward(x);
        float[][] dX = a.backward(g);

        int p = a.parameterCount();
        float[] analytic = new float[p];
        a.copyGradientsTo(analytic, 0);
        float[] params = new float[p];
        a.copyParametersTo(params, 0);
        for (int k = 0; k < p; k++) {
            float orig = params[k];
            params[k] = orig + eps;
            a.copyParametersFrom(params, 0);
            double lp = attentionLoss(a, x, g);
            params[k] = orig - eps;
            a.copyParametersFrom(params, 0);
            double lm = attentionLoss(a, x, g);
            params[k] = orig;
            double numeric = (lp - lm) / (2.0 * eps);
            assertTrue(Math.abs(analytic[k] - numeric) <= 5e-3 + 1e-2 * Math.abs(numeric),
                    "param " + k + ": analytic " + analytic[k] + " vs numeric " + numeric);
        }
        a.copyParametersFrom(params, 0);

        for (int t = 0; t < x.length; t++) {
            for (int j = 0; j < d; j++) {
                float orig = x[t][j];
                x[t][j] = orig + eps;
                double lp = attentionLoss(a, x, g);
                x[t][j] = orig - eps;
                double lm = attentionLoss(a, x, g);
                x[t][j] = orig;
                double numeric = (lp - lm) / (2.0 * eps);
                assertTrue(Math.abs(dX[t][j] - numeric) <= 5e-3 + 1e-2 * Math.abs(numeric),
                        "input [" + t + "][" + j + "]: analytic " + dX[t][j] + " vs " + numeric);
            }
        }
    }

    @Test
    void selfAttentionTrainsWithAdam() {
        SelfAttention a = new SelfAttention(4, 8L);
        float[][] x = random(5, 4, 9L);
        float[][] target = random(5, 4, 10L);
        Adam adam = new Adam(0.02f);
        double first = -1;
        double last = -1;
        for (int step = 0; step < 300; step++) {
            a.zeroGrad();
            float[][] out = a.forward(x);
            float[][] grad = new float[5][4];
            double l = 0;
            for (int t = 0; t < 5; t++) {
                for (int d = 0; d < 4; d++) {
                    float diff = out[t][d] - target[t][d];
                    grad[t][d] = diff;
                    l += 0.5 * diff * diff;
                }
            }
            if (step == 0) first = l;
            last = l;
            a.backward(grad);
            a.update(adam, 1);
        }
        assertTrue(last < 0.5 * first, "Adam did not reduce loss: " + first + " -> " + last);
    }

    @Test
    void denseClassifierExportMatchesLayerWeights() {
        DenseClassifier model = new DenseClassifier(1L, new int[] {4, 3, 2});
        WeightFile file = ModelLoader.exportDenseClassifier(model);
        Tensor w = file.tensors().get("layers.0.weight");
        assertArrayEquals(new int[] {3, 4}, w.shape());
        // Dense.weight(i, j) is input i to output j; PyTorch weight[j][i] is at j * in + i.
        float expected = model.denseLayers()[0].weight(2, 1);
        assertTrue(Math.abs(w.data()[1 * 4 + 2] - expected) < 1e-7f);
    }

    @Test
    void denseClassifierRoundTripThroughSafetensorsFile() throws Exception {
        DenseClassifier source = new DenseClassifier(11L, new int[] {6, 5, 3});
        Path file = tmp.resolve("dense.safetensors");
        SafetensorsIO.write(file, ModelLoader.exportDenseClassifier(source));

        DenseClassifier target = new DenseClassifier(99L, new int[] {6, 5, 3});
        ModelLoader.loadDenseClassifier(target, SafetensorsIO.read(file));

        float[] image = {0.1f, -0.4f, 0.9f, 0.3f, -0.2f, 0.5f};
        assertArrayEquals(source.forward(image), target.forward(image), 1e-6f);
    }

    @Test
    void convClassifierRoundTripThroughSafetensorsFile() throws Exception {
        ConvClassifier source = ConvClassifier.compact(21L, 8, 3);
        Path file = tmp.resolve("conv.safetensors");
        SafetensorsIO.write(file, ModelLoader.exportConvClassifier(source));

        ConvClassifier target = ConvClassifier.compact(77L, 8, 3);
        ModelLoader.loadConvClassifier(target, SafetensorsIO.read(file));

        float[] image = new float[64];
        for (int i = 0; i < image.length; i++) image[i] = (i % 7) / 7f;
        assertArrayEquals(source.forward(image), target.forward(image), 1e-5f);
    }

    @Test
    void missingTensorLeavesModelUnchanged() {
        DenseClassifier model = new DenseClassifier(3L, new int[] {4, 3, 2});
        float[] before = model.getParameters().clone();

        WeightFile file = ModelLoader.exportDenseClassifier(
                new DenseClassifier(4L, new int[] {4, 3, 2}));
        Map<String, Tensor> partial = new LinkedHashMap<>(file.tensors());
        partial.remove("layers.1.bias");
        WeightFile incomplete = new WeightFile(new LinkedHashMap<>(), partial);

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ModelLoader.loadDenseClassifier(model, incomplete));
        assertTrue(e.getMessage().contains("layers.1.bias"));
        assertArrayEquals(before, model.getParameters(), 0f);
    }

    @Test
    void neuralNetworkRoundTripThroughSafetensorsFile() throws Exception {
        int[] topology = {4, 5, 3};
        NeuralNetwork source = new NeuralNetwork(topology, 0.1f);
        Random r = new Random(5L);
        float[] params = new float[source.getParameters().length];
        for (int i = 0; i < params.length; i++) params[i] = (float) r.nextGaussian();
        source.setParameters(params);

        Path file = tmp.resolve("nn.safetensors");
        SafetensorsIO.write(file, ModelLoader.exportNeuralNetwork(source));

        NeuralNetwork target = new NeuralNetwork(topology, 0.1f);
        ModelLoader.loadNeuralNetwork(target, SafetensorsIO.read(file));

        float[] input = {0.2f, -0.7f, 0.4f, 1.0f};
        assertArrayEquals(source.predict(input), target.predict(input), 1e-6f);
    }

    @Test
    void neuralNetworkShapeMismatchLeavesNetworkUnchanged() {
        NeuralNetwork net = new NeuralNetwork(new int[] {4, 5, 3}, 0.1f);
        float[] before = net.getParameters().clone();
        WeightFile other = ModelLoader.exportNeuralNetwork(new NeuralNetwork(new int[] {4, 6, 3}, 0.1f));
        assertThrows(IllegalArgumentException.class,
                () -> ModelLoader.loadNeuralNetwork(net, other));
        assertArrayEquals(before, net.getParameters(), 0f);
    }

    @Test
    void wrongShapeLeavesModelUnchanged() {
        DenseClassifier model = new DenseClassifier(3L, new int[] {4, 3, 2});
        float[] before = model.getParameters().clone();

        WeightFile file = ModelLoader.exportDenseClassifier(
                new DenseClassifier(4L, new int[] {4, 5, 2}));
        assertThrows(IllegalArgumentException.class,
                () -> ModelLoader.loadDenseClassifier(model, file));
        assertArrayEquals(before, model.getParameters(), 0f);
    }
}
