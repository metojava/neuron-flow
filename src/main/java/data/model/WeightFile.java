package data.model;

import java.util.Map;

/**
 * The contents of a model weight file: string metadata plus named tensors.
 *
 * @param metadata string key/value metadata (may be empty)
 * @param tensors  tensors keyed by name, in file order
 */
public record WeightFile(Map<String, String> metadata, Map<String, Tensor> tensors) {
}
