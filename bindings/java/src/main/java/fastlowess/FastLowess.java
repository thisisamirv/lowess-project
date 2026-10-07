package fastlowess;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.Duration;

/**
 * Static utility methods for the fastlowess native library.
 */
public final class FastLowess {

    private FastLowess() {
    }

    /**
     * The released version of this Java binding (bindings/java), tracked
     * independently of the underlying fastLowess Rust core's crate version.
     */
    public static final String VERSION = "5.0.0";

    private static final String GPU_REPO = "thisisamirv/lowess-project";
    // GPU artifacts across all versions live in this one perpetual release
    // instead of cluttering each version's own release page; the source
    // version is embedded in each asset's filename instead.
    private static final String GPU_RELEASE_TAG = "gpu-builds";

    /**
     * Returns the version of this Java binding.
     *
     * @return the version string
     */
    public static String version() {
        return VERSION;
    }

    /**
     * Returns true if this library was built with the GPU execution backend
     * enabled.
     *
     * @return true if GPU support is compiled in
     */
    public static boolean gpuEnabled() {
        return NativeBridge.gpuEnabled();
    }

    /**
     * Downloads and installs a prebuilt GPU-enabled {@code fastlowess_java}
     * native library for this platform. Equivalent to
     * {@code installGpu(false)}.
     *
     * @see #installGpu(boolean)
     */
    public static void installGpu() {
        installGpu(false);
    }

    /**
     * Downloads and installs a prebuilt GPU-enabled {@code fastlowess_java}
     * native library for this platform from the matching
     * <a href="https://github.com/thisisamirv/lowess-project/releases">GitHub
     * Release</a>, saving it to {@code ~/.fastlowess/gpu/} under the standard
     * library name for this platform.
     *
     * <p>
     * A running JVM cannot swap an already-loaded native library. Point a new
     * JVM at the downloaded file via {@code -Dfastlowess.native.dir=<dir>} (or
     * {@code -Dfastlowess.native.path=<path>}) instead of relying on
     * {@code System.loadLibrary}.
     *
     * @param yes skip the interactive y/N confirmation prompt; must be true
     * when stdin is not an interactive console
     */
    public static void installGpu(boolean yes) {
        installGpu(yes, null);
    }

    /**
     * Installs a GPU-enabled {@code fastlowess_java} native library for this
     * platform, saving it to {@code ~/.fastlowess/gpu/} under the standard
     * library name for this platform.
     *
     * <p>
     * A running JVM cannot swap an already-loaded native library. Point a new
     * JVM at the installed file via {@code -Dfastlowess.native.dir=<dir>} (or
     * {@code -Dfastlowess.native.path=<path>}) instead of relying on
     * {@code System.loadLibrary}.
     *
     * @param yes skip the interactive y/N confirmation prompt; must be true
     * when stdin is not an interactive console
     * @param localPath path to a GPU-enabled library already built locally
     * (e.g. via {@code cargo build -p fastlowess-java --release --features
     * gpu}). When given, skips the GitHub Release lookup/download and installs
     * directly from this path — useful for testing the installer itself, or
     * installing an unreleased build.
     */
    public static void installGpu(boolean yes, String localPath) {
        if (gpuEnabled()) {
            System.out.println("GPU backend is already active.");
            return;
        }

        String platform = platformTag();
        String arch = archTag();
        if (platform == null || arch == null) {
            throw new IllegalStateException(
                    "No prebuilt GPU library available for " + System.getProperty("os.name")
                    + "/" + System.getProperty("os.arch") + ". Build from source instead: "
                    + "cargo build -p fastlowess-java --release --features gpu");
        }
        String ext = libraryExt(platform);
        Path dir = Paths.get(System.getProperty("user.home"), ".fastlowess", "gpu");
        Path dest = dir.resolve(libraryFileName(platform, ext));

        if (localPath != null) {
            Path src = Paths.get(localPath);
            if (!Files.isRegularFile(src)) {
                throw new IllegalStateException("No such file: " + src);
            }
            try {
                if (!containsGpuBuildMarker(src, gpuBuildMarker(platform, arch))) {
                    throw new IllegalStateException(
                            "Local library is not a GPU-enabled Java library for "
                            + platform + "/" + arch + " with ABI v4: " + src);
                }
            } catch (IOException e) {
                throw new UncheckedIOException("Failed to validate " + src, e);
            }
            if (!yes) {
                if (System.console() == null) {
                    throw new IllegalStateException(
                            "installGpu() requires confirmation. Pass yes=true to proceed non-interactively.");
                }
                String answer = System.console().readLine(
                        "Install %s in place of the current build? [y/N] ", src);
                String trimmed = answer == null ? "" : answer.strip();
                if (!"y".equalsIgnoreCase(trimmed) && !"yes".equalsIgnoreCase(trimmed)) {
                    System.out.println("Aborted.");
                    return;
                }
            }

            System.out.println("Installing " + src + " ...");
            try {
                Files.createDirectories(dir);
                Files.copy(src, dest, StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException e) {
                throw new UncheckedIOException("Failed to copy " + src, e);
            }
            System.out.println("GPU backend installed at " + dest + ".");
            System.out.println("Restart the JVM with -Dfastlowess.native.dir="
                    + dir + " for the change to take effect.");
            return;
        }

        String assetName = "libfastlowess_java-gpu-v" + VERSION + "-" + platform + "-" + arch + "." + ext;
        String url = "https://github.com/" + GPU_REPO + "/releases/download/" + GPU_RELEASE_TAG + "/" + assetName;

        if (!yes) {
            if (System.console() == null) {
                throw new IllegalStateException(
                        "installGpu() requires confirmation. Pass yes=true to proceed non-interactively.");
            }
            String answer = System.console().readLine(
                    "Download and install %s from github.com/%s? [y/N] ", assetName, GPU_REPO);
            String trimmed = answer == null ? "" : answer.strip();
            if (!"y".equalsIgnoreCase(trimmed) && !"yes".equalsIgnoreCase(trimmed)) {
                System.out.println("Aborted.");
                return;
            }
        }

        System.out.println("Downloading " + url + " ...");
        try {
            Files.createDirectories(dir);
            Path tmp = Files.createTempFile(dir, "download", ".tmp");
            try {
                HttpClient client = HttpClient.newBuilder()
                        .followRedirects(HttpClient.Redirect.NORMAL)
                        .build();
                HttpRequest request = HttpRequest.newBuilder(URI.create(url))
                        .timeout(Duration.ofMinutes(5))
                        .GET()
                        .build();
                HttpResponse<Path> response = client.send(
                        request, HttpResponse.BodyHandlers.ofFile(tmp));
                if (response.statusCode() != 200) {
                    throw new IllegalStateException(
                            "Failed to download " + url + ": HTTP " + response.statusCode()
                            + ". A matching GPU build may not exist for this platform/version yet.");
                }
                installDownloadedGpuLibrary(tmp, dest, gpuBuildMarker(platform, arch));
            } catch (IOException | InterruptedException | RuntimeException e) {
                try {
                    Files.deleteIfExists(tmp);
                } catch (IOException cleanupError) {
                    e.addSuppressed(cleanupError);
                }
                throw e;
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to download " + url, e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Download interrupted: " + url, e);
        }

        System.out.println("GPU backend installed at " + dest + ".");
        System.out.println("Restart the JVM with -Dfastlowess.native.dir="
                + dir + " for the change to take effect.");
    }

    static void installDownloadedGpuLibrary(Path tempFile, Path destination, String marker)
            throws IOException {
        if (!containsGpuBuildMarker(tempFile, marker)) {
            throw new IllegalStateException(
                    "Downloaded library is not a GPU-enabled Java library for marker " + marker);
        }
        Files.move(tempFile, destination, StandardCopyOption.REPLACE_EXISTING);
    }

    // Only these 4 platforms are built by release-gpu.yml.
    private static String platformTag() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = archTag();
        if (arch == null) {
            return null;
        }
        if (os.contains("win") && "x86_64".equals(arch)) {
            return "windows";
        }
        if ((os.contains("mac") || os.contains("darwin"))
                && ("x86_64".equals(arch) || "aarch64".equals(arch))) {
            return "macos";
        }
        if (os.contains("linux") && "x86_64".equals(arch) && !NativeBridge.isMuslLibc()) {
            return "linux";
        }
        return null;
    }

    private static String archTag() {
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (arch.contains("aarch64") || arch.contains("arm64")) {
            return "aarch64";
        }
        if (arch.contains("amd64") || arch.contains("x86_64")) {
            return "x86_64";
        }
        return null;
    }

    private static String gpuBuildMarker(String platform, String arch) {
        String libc = "linux".equals(platform) ? "-glibc" : "";
        return "fastlowess-java-gpu|abi-v4|" + platform + "-" + arch + libc;
    }

    static boolean containsGpuBuildMarker(Path archive, String marker) throws IOException {
        byte[] needle = marker.getBytes(StandardCharsets.US_ASCII);
        int chunkSize = 32 * 1024;
        byte[] buffer = new byte[chunkSize + needle.length - 1];
        int carry = 0;
        try (InputStream input = Files.newInputStream(archive)) {
            while (true) {
                int count = input.read(buffer, carry, chunkSize);
                if (count < 0) {
                    return false;
                }
                if (count == 0) {
                    continue;
                }
                int length = carry + count;
                if (containsBytes(buffer, length, needle)) {
                    return true;
                }
                carry = Math.min(needle.length - 1, length);
                System.arraycopy(buffer, length - carry, buffer, 0, carry);
            }
        }
    }

    private static boolean containsBytes(byte[] data, int length, byte[] needle) {
        for (int offset = 0; offset <= length - needle.length; offset++) {
            int index = 0;
            while (index < needle.length && data[offset + index] == needle[index]) {
                index++;
            }
            if (index == needle.length) {
                return true;
            }
        }
        return false;
    }

    private static String libraryExt(String platform) {
        return switch (platform) {
            case "windows" ->
                "dll";
            case "macos" ->
                "dylib";
            default ->
                "so";
        };
    }

    private static String libraryFileName(String platform, String ext) {
        return "windows".equals(platform) ? "fastlowess_java." + ext : "libfastlowess_java." + ext;
    }
}
