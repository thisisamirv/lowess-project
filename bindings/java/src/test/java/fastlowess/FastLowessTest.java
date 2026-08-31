package fastlowess;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import org.junit.jupiter.api.Test;

class FastLowessTest {

    @Test
    void reportsVersion() {
        assertNotNull(FastLowess.version());
        assertFalse(FastLowess.version().isBlank());
    }

    @Test
    void reportsGpuEnabled() {
        // The debug test build is compiled without the `gpu` feature.
        assertFalse(FastLowess.gpuEnabled());
    }
}
