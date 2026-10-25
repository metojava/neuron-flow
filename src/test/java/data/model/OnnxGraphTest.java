package data.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class OnnxGraphTest {

    @TempDir
    Path tmp;

    private static OnnxGraph sampleGraph() {
        Map<String, Object> gemm = new LinkedHashMap<>();
        gemm.put("alpha", 1.5f);
        gemm.put("transB", 1L);
        gemm.put("kernel_shape", List.of(3L, 3L));
        gemm.put("mode", "constant");
        gemm.put("scales", List.of(0.5f, 2f));
        gemm.put("names", List.of("a", "b"));
        OnnxGraph.Node node = new OnnxGraph.Node("fc", "Gemm",
                List.of("x", "fc.weight", "fc.bias"), List.of("y"), gemm);
        OnnxGraph.Node relu = new OnnxGraph.Node("", "Relu",
                List.of("y"), List.of("z"), new LinkedHashMap<>());
        return new OnnxGraph("tiny", List.of(new OnnxGraph.ValueInfo("x", 1, new long[] {-1, 4})),
                List.of(new OnnxGraph.ValueInfo("z", 1, new long[] {-1, 3})),
                List.of(node, relu), List.of());
    }

    private static WeightFile sampleWeights() {
        Map<String, Tensor> t = new LinkedHashMap<>();
        t.put("fc.weight", new Tensor("fc.weight", new int[] {3, 4}, new float[12]));
        t.put("fc.bias", new Tensor("fc.bias", new int[] {3}, new float[] {0.1f, 0.2f, 0.3f}));
        return new WeightFile(new LinkedHashMap<>(), t);
    }

    @Test
    void graphStructureRoundTrips() throws IOException {
        Path file = tmp.resolve("graph.onnx");
        OnnxIO.writeGraph(file, sampleGraph(), sampleWeights());

        try (OnnxIO.Reader r = new OnnxIO.Reader(file)) {
            OnnxGraph g = r.graph();
            assertEquals("tiny", g.name());

            assertEquals(1, g.inputs().size());
            assertEquals("x", g.inputs().get(0).name());
            assertEquals(1, g.inputs().get(0).elemType());
            assertArrayEquals(new long[] {-1, 4}, g.inputs().get(0).shape());
            assertArrayEquals(new long[] {-1, 3}, g.outputs().get(0).shape());

            assertEquals(2, g.nodes().size());
            OnnxGraph.Node gemm = g.nodes().get(0);
            assertEquals("fc", gemm.name());
            assertEquals("Gemm", gemm.opType());
            assertEquals(List.of("x", "fc.weight", "fc.bias"), gemm.inputs());
            assertEquals(List.of("y"), gemm.outputs());
            assertEquals(1.5f, gemm.attributes().get("alpha"));
            assertEquals(1L, gemm.attributes().get("transB"));
            assertEquals(List.of(3L, 3L), gemm.attributes().get("kernel_shape"));
            assertEquals("constant", gemm.attributes().get("mode"));
            assertEquals(List.of(0.5f, 2f), gemm.attributes().get("scales"));
            assertEquals(List.of("a", "b"), gemm.attributes().get("names"));

            OnnxGraph.Node relu = g.nodes().get(1);
            assertEquals("", relu.name());
            assertEquals("Relu", relu.opType());
            assertTrue(relu.attributes().isEmpty());
        }
    }

    @Test
    void initializersAreListedAndStillLoad() throws IOException {
        Path file = tmp.resolve("with-weights.onnx");
        OnnxIO.writeGraph(file, sampleGraph(), sampleWeights());
        try (OnnxIO.Reader r = new OnnxIO.Reader(file)) {
            assertEquals(List.of("fc.weight", "fc.bias"), r.graph().initializerNames());
            assertArrayEquals(new float[] {0.1f, 0.2f, 0.3f}, r.load("fc.bias").data(), 0f);
        }
    }

    @Test
    void packedIntsAndTensorAttributesAreHandled() throws IOException {
        // Attribute 'pads' with packed INTS (wire type 2), and a tensor attribute 't' that is
        // recorded as null.
        OnnxIO.Proto.Writer pads = new OnnxIO.Proto.Writer();
        pads.string(1, "pads");
        pads.varint(20, 7);
        pads.bytes(8, new byte[] {1, 2, 3});

        OnnxIO.Proto.Writer tensorAttr = new OnnxIO.Proto.Writer();
        tensorAttr.string(1, "value");
        tensorAttr.varint(20, 4);
        tensorAttr.bytes(5, new byte[] {});

        OnnxIO.Proto.Writer node = new OnnxIO.Proto.Writer();
        node.string(4, "Pad");
        node.bytes(5, pads.toByteArray());
        node.bytes(5, tensorAttr.toByteArray());

        OnnxIO.Proto.Writer graph = new OnnxIO.Proto.Writer();
        graph.bytes(1, node.toByteArray());
        OnnxIO.Proto.Writer model = new OnnxIO.Proto.Writer();
        model.bytes(7, graph.toByteArray());
        Path file = tmp.resolve("packed.onnx");
        Files.write(file, model.toByteArray());

        try (OnnxIO.Reader r = new OnnxIO.Reader(file)) {
            OnnxGraph.Node n = r.graph().nodes().get(0);
            assertEquals(List.of(1L, 2L, 3L), n.attributes().get("pads"));
            assertTrue(n.attributes().containsKey("value"));
            assertNull(n.attributes().get("value"));
        }
    }

    @Test
    void weightsOnlyFilesHaveAnEmptyGraph() throws IOException {
        Path file = tmp.resolve("weights.onnx");
        OnnxIO.write(file, sampleWeights());
        try (OnnxIO.Reader r = new OnnxIO.Reader(file)) {
            OnnxGraph g = r.graph();
            assertTrue(g.nodes().isEmpty());
            assertTrue(g.inputs().isEmpty());
            assertEquals(2, g.initializerNames().size());
        }
    }
}
