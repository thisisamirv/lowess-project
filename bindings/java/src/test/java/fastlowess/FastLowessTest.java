package fastlowess;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
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

    @Test
    void validatesGpuBuildMarkerInLocalArchive() throws IOException {
        String marker = "fastlowess-java-gpu|abi-v4|windows-x86_64";
        Path archive = Files.createTempFile("fastlowess-gpu-marker", ".bin");
        try {
            byte[] content = new byte[32 * 1024 + marker.length()];
            System.arraycopy(marker.getBytes(StandardCharsets.US_ASCII), 0, content, 32 * 1024, marker.length());
            Files.write(archive, content);
            assertTrue(FastLowess.containsGpuBuildMarker(archive, marker));
            assertFalse(FastLowess.containsGpuBuildMarker(archive, "fastlowess-java-gpu|abi-v4|linux-x86_64-glibc"));
        } finally {
            Files.deleteIfExists(archive);
        }

        Path invalid = Files.createTempFile("fastlowess-gpu-invalid", ".bin");
        try {
            Files.writeString(invalid, "not a GPU library");
            assertFalse(FastLowess.containsGpuBuildMarker(invalid, marker));
        } finally {
            Files.deleteIfExists(invalid);
        }
    }
}
