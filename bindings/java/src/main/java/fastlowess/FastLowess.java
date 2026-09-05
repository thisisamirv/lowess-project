package fastlowess;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
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
    public static final String VERSION = "4.0.0";

    private static final String GPU_REPO = "thisisamirv/lowess-project";

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
        if (gpuEnabled()) {
            System.out.println("GPU backend is already active.");
            return;
        }

        String platform = platformTag();
        String arch = archTag();
        if (platform == null) {
            throw new IllegalStateException(
                    "No prebuilt GPU library available for " + System.getProperty("os.name")
                    + "/" + System.getProperty("os.arch") + ". Build from source instead: "
                    + "cargo build -p fastlowess-java --release --features gpu");
        }
        String ext = libraryExt(platform);
        String assetName = "libfastlowess_java-gpu-v" + VERSION + "-" + platform + "-" + arch + "." + ext;
        String url = "https://github.com/" + GPU_REPO + "/releases/download/v" + VERSION + "/" + assetName;

        if (!yes) {
            if (System.console() == null) {
                throw new IllegalStateException(
                        "installGpu() requires confirmation. Pass yes=true to proceed non-interactively.");
            }
            String answer = System.console().readLine(
                    "Download and install %s from github.com/%s? [y/N] ", assetName, GPU_REPO);
            String trimmed = answer == null ? "" : answer.strip();
            if (!trimmed.equalsIgnoreCase("y") && !trimmed.equalsIgnoreCase("yes")) {
                System.out.println("Aborted.");
                return;
            }
        }

        Path dir = Paths.get(System.getProperty("user.home"), ".fastlowess", "gpu");
        Path dest = dir.resolve(libraryFileName(platform, ext));
        System.out.println("Downloading " + url + " ...");
        try {
            Files.createDirectories(dir);
            Path tmp = Files.createTempFile(dir, "download", ".tmp");
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
                Files.deleteIfExists(tmp);
                throw new IllegalStateException(
                        "Failed to download " + url + ": HTTP " + response.statusCode()
                        + ". A matching GPU build may not exist for this platform/version yet.");
            }
            Files.move(tmp, dest, StandardCopyOption.REPLACE_EXISTING);
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

    // Only these 4 platforms are built by release-gpu.yml.
    private static String platformTag() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        boolean isArm = arch.contains("aarch64") || arch.contains("arm64");
        if (os.contains("win")) {
            return isArm ? null : "windows";
        }
        if (os.contains("mac") || os.contains("darwin")) {
            return "macos";
        }
        if (os.contains("linux")) {
            return isArm ? null : "linux";
        }
        return null;
    }

    private static String archTag() {
        String arch = System.getProperty("os.arch", "").toLowerCase();
        return (arch.contains("aarch64") || arch.contains("arm64")) ? "aarch64" : "x86_64";
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
        return platform.equals("windows") ? "fastlowess_java." + ext : "libfastlowess_java." + ext;
    }
}
