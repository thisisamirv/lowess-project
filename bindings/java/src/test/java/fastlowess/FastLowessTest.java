package fastlowess;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
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

    @Test
    void validatesDownloadedGpuLibraryBeforePromotion() throws IOException {
        String marker = "fastlowess-java-gpu|abi-v4|windows-x86_64";
        Path directory = Files.createTempDirectory("fastlowess-gpu-download");
        Path destination = directory.resolve("fastlowess_java.dll");
        Path wrongTarget = Files.createTempFile(directory, "wrong-target", ".tmp");
        Files.writeString(wrongTarget, "fastlowess-java-gpu|abi-v4|linux-x86_64-glibc");

        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> FastLowess.installDownloadedGpuLibrary(wrongTarget, destination, marker));
        assertTrue(error.getMessage().contains(marker));
        assertFalse(Files.exists(destination));

        Path matchingTarget = Files.createTempFile(directory, "matching-target", ".tmp");
        Files.writeString(matchingTarget, marker);
        FastLowess.installDownloadedGpuLibrary(matchingTarget, destination, marker);
        assertTrue(Files.exists(destination));
        assertFalse(Files.exists(matchingTarget));
    }
}
