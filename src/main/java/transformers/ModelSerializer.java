package transformers;

import autograd.Tensor;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Utility for saving and loading transformer models with config and weights.
 * Uses a simple directory-based format: config.json + weights.bin
 */
public final class ModelSerializer {

    private ModelSerializer() {}

    /**
     * Saves a model (config + weights) to a directory.
     */
    public static void save(Object model, TransformerConfig config, Path outputDir) throws IOException {
        Files.createDirectories(outputDir);

        // Save config with class name
        Path configPath = outputDir.resolve("config.json");
        try (BufferedWriter writer = Files.newBufferedWriter(configPath)) {
            writer.write(configToJson(config, config.getClass().getName()));
        }

        // Save weights
        Path weightsPath = outputDir.resolve("weights.bin");
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(weightsPath)))) {

            List<Tensor> params = getParameters(model);
            out.writeInt(params.size());
            for (Tensor param : params) {
                float[] data = param.toFlat();
                int[] shape = param.shape();
                out.writeInt(shape.length);
                for (int dim : shape) out.writeInt(dim);
                out.writeInt(data.length);
                for (float v : data) out.writeFloat(v);
            }
        }
    }

    /**
     * Loads a model from a directory.
     */
    public static Object load(Class<?> modelClass, Class<? extends TransformerConfig> configClass, Path inputDir) throws Exception {
        // Load config JSON
        String json = Files.readString(inputDir.resolve("config.json"));
        System.out.println("Loaded JSON: " + json);
        Map<String, Object> map = jsonToMap(json);
        System.out.println("Parsed map: " + map);

        // Read class name from JSON, fallback to provided configClass
        String className = (String) map.remove("_class");
        Class<? extends TransformerConfig> actualConfigClass = configClass;
        if (className != null) {
            try {
                actualConfigClass = (Class<? extends TransformerConfig>) Class.forName(className);
            } catch (ClassNotFoundException ignored) {
                // fallback to provided configClass
            }
        }

        // Instantiate the specific config class and populate from map
        TransformerConfig config = actualConfigClass.getDeclaredConstructor().newInstance();
        for (Map.Entry<String, Object> e : map.entrySet()) {
            String key = e.getKey();
            Object val = e.getValue();
            String setterName = "set" + toCamelCase(key);
            System.out.println("Setting " + setterName + " = " + val + " (type: " + val.getClass().getName() + ")");
            try {
                java.lang.reflect.Method setter = actualConfigClass.getMethod(setterName, val.getClass());
                setter.invoke(config, val);
                System.out.println("  Set via " + val.getClass().getName());
            } catch (NoSuchMethodException ignored0) {
                // Try primitive types
                try {
                    if (val instanceof Integer) {
                        actualConfigClass.getMethod("set" + toCamelCase(key), int.class).invoke(config, val);
                        System.out.println("  Set via int");
                    } else if (val instanceof Float) {
                        actualConfigClass.getMethod("set" + toCamelCase(key), float.class).invoke(config, val);
                        System.out.println("  Set via float");
                    } else if (val instanceof String) {
                        actualConfigClass.getMethod("set" + toCamelCase(key), String.class).invoke(config, val);
                        System.out.println("  Set via String");
                    }
                } catch (NoSuchMethodException ignored1) {}
            }
        }
        System.out.println("Final config: vocabSize=" + config.getVocabSize() + ", hiddenSize=" + config.getHiddenSize() + 
            ", numHiddenLayers=" + config.getNumHiddenLayers() + ", numAttentionHeads=" + config.getNumAttentionHeads() +
            ", intermediateSize=" + config.getIntermediateSize() + ", maxPositionEmbeddings=" + config.getMaxPositionEmbeddings());

        // Instantiate model via reflection
        Object model;
        System.out.println("Loading model class: " + modelClass.getName() + " with config class: " + actualConfigClass.getName());
        System.out.println("Config: vocabSize=" + config.getVocabSize() + ", hiddenSize=" + config.getHiddenSize() + 
            ", numHiddenLayers=" + config.getNumHiddenLayers() + ", numAttentionHeads=" + config.getNumAttentionHeads() +
            ", intermediateSize=" + config.getIntermediateSize() + ", maxPositionEmbeddings=" + config.getMaxPositionEmbeddings());
        try {
            model = modelClass.getConstructor(actualConfigClass, long.class).newInstance(config, 42L);
        } catch (NoSuchMethodException e) {
            model = modelClass.getConstructor(actualConfigClass).newInstance(config);
        }
        System.out.println("Created model: " + model.getClass().getName() + " with " + getParameters(model).size() + " parameters");

        // Load weights
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(inputDir.resolve("weights.bin"))))) {

            int numParams = in.readInt();
            List<Tensor> params = getParameters(modelClass.cast(model));
            if (params.size() != numParams) {
                throw new IllegalStateException("Parameter count mismatch: expected " + numParams + ", model has " + params.size());
            }

            for (int i = 0; i < numParams; i++) {
                int ndim = in.readInt();
                int[] shape = new int[ndim];
                for (int d = 0; d < ndim; d++) shape[d] = in.readInt();
                int length = in.readInt();
                float[] data = new float[length];
                for (int j = 0; j < length; j++) data[j] = in.readFloat();

                Tensor param = params.get(i);
                if (param.numel() != length) {
                    throw new IllegalStateException("Parameter " + i + " size mismatch: expected " + param.numel() + ", got " + length);
                }
                float[] paramData = param.toFlat();
                System.arraycopy(data, 0, paramData, 0, length);
            }
        }

        return model;
    }

    // --- Helpers ---

    private static List<Tensor> getParameters(Object model) {
        try {
            java.lang.reflect.Method m = model.getClass().getMethod("parameters");
            @SuppressWarnings("unchecked")
            List<Tensor> params = (List<Tensor>) m.invoke(model);
            return params;
        } catch (Exception e) {
            throw new IllegalStateException("Model must have parameters() method returning List<Tensor>", e);
        }
    }

    private static String configToJson(TransformerConfig config, String className) {
        Map<String, Object> map = config.toMap();
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        boolean first = true;
        for (Map.Entry<String, Object> e : map.entrySet()) {
            if (!first) sb.append(",\n");
            first = false;
            sb.append("  \"").append(e.getKey()).append("\": ");
            Object v = e.getValue();
            if (v instanceof String) sb.append("\"").append(v).append("\"");
            else sb.append(v);
        }
        sb.append(",\n  \"_class\": \"").append(className).append("\"\n}");
        return sb.toString();
    }

    private static String configToJson(TransformerConfig config) {
        return configToJson(config, config.getClass().getName());
    }

    private static Map<String, Object> jsonToMap(String json) {
        Map<String, Object> map = new HashMap<>();
        String content = json.trim();
        if (content.startsWith("{")) content = content.substring(1);
        if (content.endsWith("}")) content = content.substring(0, content.length() - 1);

        String[] pairs = content.split(",");
        for (String pair : pairs) {
            String[] kv = pair.split(":", 2);
            if (kv.length != 2) continue;
            String key = kv[0].trim().replace("\"", "").trim();
            String val = kv[1].trim();
            if (val.startsWith("\"") && val.endsWith("\"")) {
                map.put(key, val.substring(1, val.length() - 1));
            } else if (val.contains(".")) {
                map.put(key, Float.parseFloat(val));
            } else {
                map.put(key, Integer.parseInt(val));
            }
        }
        return map;
    }

    private static String toCamelCase(String snake) {
        String[] parts = snake.split("_");
        StringBuilder sb = new StringBuilder();
        for (String part : parts) {
            if (part.length() > 0) {
                sb.append(Character.toUpperCase(part.charAt(0)));
                if (part.length() > 1) sb.append(part.substring(1));
            }
        }
        return sb.toString();
    }
}