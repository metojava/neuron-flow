package data.model;

import java.util.List;
import java.util.Map;

/**
 * Structure of an ONNX graph: inputs, outputs, nodes and the names of its initializers.
 * Initializer values are read separately through {@link OnnxIO.Reader#load}.
 *
 * @param name             graph name (empty if absent)
 * @param inputs           graph inputs in file order
 * @param outputs          graph outputs in file order
 * @param nodes            nodes in file order (ONNX stores them topologically sorted)
 * @param initializerNames names of the constant tensors stored with the graph
 */
public record OnnxGraph(String name, List<ValueInfo> inputs, List<ValueInfo> outputs,
                        List<Node> nodes, List<String> initializerNames) {

    /**
     * Name, element type and shape of a graph value. A dimension of -1 is symbolic or
     * unknown. A value with no shape information has an empty shape array.
     *
     * @param elemType ONNX TensorProto.DataType number (1 = FLOAT, 7 = INT64, ...)
     */
    public record ValueInfo(String name, int elemType, long[] shape) {
    }

    /**
     * One operator invocation.
     *
     * <p>Attribute values are {@code Long} (INT), {@code Float} (FLOAT), {@code String}
     * (STRING), or {@code List} of Long, Float or String (INTS, FLOATS, STRINGS). Attributes
     * of kind TENSOR, GRAPH or SPARSE_TENSOR are recorded with a {@code null} value.
     */
    public record Node(String name, String opType, List<String> inputs, List<String> outputs,
                       Map<String, Object> attributes) {
    }
}
