package transformers;

import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.nio.file.*;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * Safetensors format loader.
 * Loads tensors from .safetensors files (used by Hugging Face).
 * Format: header (8 bytes length + JSON) + tensor data
 */
public final class SafetensorsLoader {

    private SafetensorsLoader() {}

    /**
     * Loads all tensors from a safetensors file.
     *
     * @param path path to .safetensors file
     * @return map of tensor name -> Tensor
     */
    public static Map<String, autograd.Tensor> load(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            // Read header length (8 bytes, little-endian)
            ByteBuffer headerLenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(headerLenBuf);
            headerLenBuf.flip();
            long headerLen = headerLenBuf.getLong();

            // Read header JSON
            ByteBuffer headerBuf = ByteBuffer.allocate((int) headerLen);
            channel.read(headerBuf);
            headerBuf.flip();
            byte[] headerBytes = new byte[(int) headerLen];
            headerBuf.get(headerBytes);
            String headerJson = new String(headerBytes, java.nio.charset.StandardCharsets.UTF_8);

            // Parse header JSON (simple parser for flat key-value)
            Map<String, TensorInfo> tensorInfoMap = parseHeader(headerJson);

            // Read tensor data
            Map<String, autograd.Tensor> tensors = new HashMap<>();
            for (Map.Entry<String, TensorInfo> e : tensorInfoMap.entrySet()) {
                String name = e.getKey();
                TensorInfo info = e.getValue();

                // Read tensor data
                channel.position(info.dataOffsets[0]);
                ByteBuffer dataBuf = ByteBuffer.allocate(info.numBytes).order(ByteOrder.LITTLE_ENDIAN);
                channel.read(dataBuf);
                dataBuf.flip();

                // Convert to float array based on dtype
                float[] data = convertToFloat(dataBuf, info.dtype, info.numElements);

                autograd.Tensor tensor = autograd.Tensor.of(data, info.shape);
                tensors.put(name, tensor);
            }

            return tensors;
        }
    }

    private static class TensorInfo {
        String dtype;
        int[] shape;
        long[] dataOffsets;
        int numElements;
        int numBytes;
    }

    private static Map<String, TensorInfo> parseHeader(String json) {
        Map<String, TensorInfo> map = new HashMap<>();
        // Simple JSON parser for the tensor metadata
        // Format: {"tensor_name": {"dtype": "F32", "shape": [1,2,3], "data_offsets": [0, 100]}, ...}
        // This is a simplified parser - in production use a proper JSON library
        String content = json.trim();
        if (content.startsWith("{")) content = content.substring(1);
        if (content.endsWith("}")) content = content.substring(0, content.length() - 1);

        // Split by }, (but not inside nested braces)
        int depth = 0;
        StringBuilder current = new StringBuilder();
        List<String> entries = new ArrayList<>();
        for (char c : content.toCharArray()) {
            current.append(c);
            if (c == '{') depth++;
            else if (c == '}') depth--;
            if (depth == 0 && c == '}') {
                entries.add(current.toString());
                current = new StringBuilder();
            }
        }

        for (String entry : entries) {
            // Parse each tensor entry
            // Format: "name": {dtype: "...", shape: [...], data_offsets: [...]}
            int colonIdx = entry.indexOf(':');
            if (colonIdx == -1) continue;
            String name = entry.substring(0, colonIdx).trim().replace("\"", "");
            String obj = entry.substring(colonIdx + 1).trim();

            TensorInfo info = new TensorInfo();
            // Parse dtype
            int dtypeIdx = obj.indexOf("dtype");
            if (dtypeIdx >= 0) {
                int start = obj.indexOf('"', dtypeIdx + 6);
                int end = obj.indexOf('"', start + 1);
                info.dtype = obj.substring(start + 1, end);
            }
            // Parse shape
            int shapeIdx = obj.indexOf("shape");
            if (shapeIdx >= 0) {
                int start = obj.indexOf('[', shapeIdx);
                int end = obj.indexOf(']', start);
                String shapeStr = obj.substring(start + 1, end);
                String[] parts = shapeStr.split(",");
                info.shape = new int[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    info.shape[i] = Integer.parseInt(parts[i].trim());
                }
                info.numElements = 1;
                for (int s : info.shape) info.numElements *= s;
            }
            // Parse data_offsets
            int offsetIdx = obj.indexOf("data_offsets");
            if (offsetIdx >= 0) {
                int start = obj.indexOf('[', offsetIdx);
                int end = obj.indexOf(']', start);
                String offsetStr = obj.substring(start + 1, end);
                String[] parts = offsetStr.split(",");
                info.dataOffsets = new long[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    info.dataOffsets[i] = Long.parseLong(parts[i].trim());
                }
            }
            // Calculate numBytes
            int elementSize = dtypeSize(info.dtype);
            info.numBytes = info.numElements * elementSize;

            map.put(name, info);
        }

        return map;
    }

    private static int dtypeSize(String dtype) {
        if ("F32".equals(dtype) || "F64".equals(dtype)) return 4;
        if ("I32".equals(dtype) || "I64".equals(dtype)) return 4;
        if ("F16".equals(dtype) || "BF16".equals(dtype)) return 2;
        if ("I8".equals(dtype) || "U8".equals(dtype) || "BOOL".equals(dtype)) return 1;
        return 4; // default
    }

    private static float[] convertToFloat(ByteBuffer buf, String dtype, int numElements) {
        float[] out = new float[numElements];
        if ("F32".equals(dtype)) {
            buf.asFloatBuffer().get(out);
        } else if ("F16".equals(dtype) || "BF16".equals(dtype)) {
            // Half precision to float
            ShortBuffer sb = buf.asShortBuffer();
            for (int i = 0; i < numElements; i++) {
                out[i] = halfToFloat(sb.get(i));
            }
        } else if ("F64".equals(dtype)) {
            DoubleBuffer db = buf.asDoubleBuffer();
            for (int i = 0; i < numElements; i++) {
                out[i] = (float) db.get(i);
            }
        } else if ("I32".equals(dtype) || "I64".equals(dtype)) {
            IntBuffer ib = buf.asIntBuffer();
            for (int i = 0; i < numElements; i++) {
                out[i] = ib.get(i);
            }
        } else {
            // Default: treat as float32
            buf.asFloatBuffer().get(out);
        }
        return out;
    }

    private static float halfToFloat(short h) {
        // IEEE 754 half-precision to float
        int sign = (h >> 15) & 0x1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;

        if (exp == 0) {
            if (mant == 0) return sign == 1 ? -0.0f : 0.0f;
            // Subnormal
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x400;
        } else if (exp == 31) {
            return mant == 0 ? (sign == 1 ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY) : Float.NaN;
        }

        int fExp = exp + (127 - 15);
        int fMant = mant << 13;
        int fBits = (sign << 31) | (fExp << 23) | fMant;
        return Float.intBitsToFloat(fBits);
    }
}