package activations;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ActivationTest {
    @Test
    void geluDerivWithZ() {
        ActivationFunction gelu = ActivationFunction.gelu();
        float x = 1.2f;
        float a = gelu.activate(x);
        float d = gelu.derivative(x, a);
        assertTrue(Float.isFinite(d));
    }

    @Test
    void swishDerivWithZ() {
        ActivationFunction swish = ActivationFunction.swish();
        float x = -0.5f;
        float a = swish.activate(x);
        float d = swish.derivative(x, a);
        assertTrue(Float.isFinite(d));
    }

    @Test
    void relu6() {
        ReLU6 r = ReLU6.of();
        assertEquals(0.0f, r.activate(-1.0f), 1e-6);
        assertEquals(3.0f, r.activate(3.0f), 1e-6);
        assertEquals(6.0f, r.activate(7.0f), 1e-6);
    }

    @Test
    void mishFinite() {
        Mish m = Mish.of();
        float a = m.activate(0.5f);
        assertTrue(Float.isFinite(a));
    }
}
