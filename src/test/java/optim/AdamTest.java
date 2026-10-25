package optim;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AdamTest {
    @Test
    void firstStepUsesBiasCorrection() {
        Adam adam = new Adam(0.01f);
        float[] parameters = {1.0f, -1.0f};
        adam.update(parameters, new float[] {2.0f, -2.0f});
        assertArrayEquals(new float[] {0.99f, -0.99f}, parameters, 1e-6f);
    }

    @Test
    void rejectsInvalidConfigurationAndShape() {
        assertThrows(IllegalArgumentException.class, () -> new Adam(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> new Adam(0.01f, 1.0f, 0.999f, 1e-8f));
        assertThrows(IllegalArgumentException.class,
            () -> new Adam(0.01f).update(new float[] {1}, new float[] {1, 2}));
    }
}
