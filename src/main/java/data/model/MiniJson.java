package data.model;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Small JSON parser for safetensors headers. Objects become LinkedHashMap,
 * arrays become List, numbers become Double, and strings, booleans and null
 * map to their Java equivalents.
 */
final class MiniJson {

    private final String s;
    private int pos;

    private MiniJson(String s) {
        this.s = s;
    }

    static Object parse(String text) throws IOException {
        MiniJson p = new MiniJson(text);
        Object v = p.value();
        p.ws();
        if (p.pos != p.s.length()) throw p.err("trailing data");
        return v;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> asMap(Object o) throws IOException {
        if (o instanceof Map) return (Map<String, Object>) o;
        throw new IOException("Expected JSON object");
    }

    /** Quotes and escapes a string as a JSON string literal. */
    static String quote(String str) {
        StringBuilder sb = new StringBuilder(str.length() + 2).append('"');
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        return sb.append('"').toString();
    }

    private Object value() throws IOException {
        ws();
        if (pos >= s.length()) throw err("unexpected end of input");
        char c = s.charAt(pos);
        switch (c) {
            case '{': return object();
            case '[': return array();
            case '"': return string();
            case 't': expect("true"); return Boolean.TRUE;
            case 'f': expect("false"); return Boolean.FALSE;
            case 'n': expect("null"); return null;
            default: return number();
        }
    }

    private Map<String, Object> object() throws IOException {
        pos++;
        Map<String, Object> m = new LinkedHashMap<>();
        ws();
        if (peek() == '}') {
            pos++;
            return m;
        }
        while (true) {
            ws();
            String key = string();
            ws();
            expectChar(':');
            m.put(key, value());
            ws();
            char c = next();
            if (c == '}') return m;
            if (c != ',') throw err("expected ',' or '}'");
        }
    }

    private List<Object> array() throws IOException {
        pos++;
        List<Object> list = new ArrayList<>();
        ws();
        if (peek() == ']') {
            pos++;
            return list;
        }
        while (true) {
            list.add(value());
            ws();
            char c = next();
            if (c == ']') return list;
            if (c != ',') throw err("expected ',' or ']'");
        }
    }

    private String string() throws IOException {
        expectChar('"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            if (pos >= s.length()) throw err("unterminated string");
            char c = s.charAt(pos++);
            if (c == '"') return sb.toString();
            if (c != '\\') {
                sb.append(c);
                continue;
            }
            char e = next();
            switch (e) {
                case '"': case '\\': case '/': sb.append(e); break;
                case 'b': sb.append('\b'); break;
                case 'f': sb.append('\f'); break;
                case 'n': sb.append('\n'); break;
                case 'r': sb.append('\r'); break;
                case 't': sb.append('\t'); break;
                case 'u':
                    if (pos + 4 > s.length()) throw err("bad unicode escape");
                    sb.append((char) Integer.parseInt(s.substring(pos, pos + 4), 16));
                    pos += 4;
                    break;
                default: throw err("bad escape");
            }
        }
    }

    private Double number() throws IOException {
        int start = pos;
        while (pos < s.length() && "+-0123456789.eE".indexOf(s.charAt(pos)) >= 0) pos++;
        if (start == pos) throw err("unexpected character");
        try {
            return Double.parseDouble(s.substring(start, pos));
        } catch (NumberFormatException e) {
            throw err("bad number");
        }
    }

    private void ws() {
        while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) pos++;
    }

    private char peek() {
        return pos < s.length() ? s.charAt(pos) : '\0';
    }

    private char next() throws IOException {
        if (pos >= s.length()) throw err("unexpected end of input");
        return s.charAt(pos++);
    }

    private void expect(String literal) throws IOException {
        if (!s.startsWith(literal, pos)) throw err("expected " + literal);
        pos += literal.length();
    }

    private void expectChar(char c) throws IOException {
        if (next() != c) throw err("expected '" + c + "'");
    }

    private IOException err(String msg) {
        return new IOException("JSON: " + msg + " at offset " + pos);
    }
}
