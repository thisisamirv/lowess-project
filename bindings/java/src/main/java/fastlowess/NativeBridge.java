package fastlowess;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Declares the JNI native methods implemented by the Rust
 * {@code fastlowess_java} crate, and loads the platform-specific native library
 * backing them.
 */
final class NativeBridge {

    private NativeBridge() {
    }

    static {
        loadNativeLibrary();
    }

    private static void loadNativeLibrary() {
        String explicitPath = System.getProperty("fastlowess.native.path");
        if (explicitPath != null) {
            System.load(explicitPath);
            return;
        }

        String nativeDir = System.getProperty("fastlowess.native.dir");
        if (nativeDir != null) {
            File candidate = new File(nativeDir, mapLibraryName());
            if (candidate.isFile()) {
                System.load(candidate.getAbsolutePath());
                return;
            }
        }

        try {
            System.loadLibrary("fastlowess_java");
            return;
        } catch (UnsatisfiedLinkError ignored) {
            // Fall through to the bundled-resource lookup below.
        }

        loadFromBundledResource();
    }

    // Extracts a platform-specific native library bundled inside the JAR (under
    // /native/<os>-<arch>/<libname>) to a temp file, then loads it.
    private static void loadFromBundledResource() {
        String resourcePath = "/native/" + osArchDir() + "/" + mapLibraryName();
        try (InputStream in = NativeBridge.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new UnsatisfiedLinkError(
                        "Could not locate the fastlowess native library (looked for classpath "
                        + "resource '" + resourcePath + "', system property "
                        + "'fastlowess.native.path', 'fastlowess.native.dir', and "
                        + "java.library.path). Build it with `cargo build -p fastlowess-java`.");
            }
            Path tempFile = Files.createTempFile("fastlowess_java", suffix());
            tempFile.toFile().deleteOnExit();
            Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
            System.load(tempFile.toAbsolutePath().toString());
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to extract the fastlowess native library", e);
        }
    }

    private static String mapLibraryName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            return "fastlowess_java.dll";
        }
        if (os.contains("mac") || os.contains("darwin")) {
            return "libfastlowess_java.dylib";
        }
        return "libfastlowess_java.so";
    }

    private static String suffix() {
        String name = mapLibraryName();
        int dot = name.lastIndexOf('.');
        return dot >= 0 ? name.substring(dot) : "";
    }

    private static String osArchDir() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        String osName = os.contains("win") ? "windows" : (os.contains("mac") ? "macos" : "linux");
        String archName = (arch.contains("aarch64") || arch.contains("arm64")) ? "aarch64" : "x86_64";
        return osName + "-" + archName;
    }

    static native String version();

    static native boolean gpuEnabled();

    static native long lowessNew(
            double fraction,
            int iterations,
            double delta,
            String weightFunction,
            String robustnessMethod,
            String scalingMethod,
            String boundaryPolicy,
            double confidenceIntervals,
            double predictionIntervals,
            boolean returnDiagnostics,
            boolean returnResiduals,
            boolean returnRobustnessWeights,
            String zeroWeightFallback,
            double autoConverge,
            double[] cvFractions,
            String cvMethod,
            int cvK,
            boolean parallel,
            boolean returnSe,
            String backend);

    static native void lowessSetCvSeed(long handle, long seed);

    static native NativeResult lowessFit(long handle, double[] x, double[] y, double[] customWeights);

    static native void lowessFree(long handle);

    static native long streamingNew(
            double fraction,
            int iterations,
            double delta,
            String weightFunction,
            String robustnessMethod,
            String scalingMethod,
            String boundaryPolicy,
            boolean returnDiagnostics,
            boolean returnResiduals,
            boolean returnRobustnessWeights,
            String zeroWeightFallback,
            double autoConverge,
            boolean parallel,
            int chunkSize,
            int overlap,
            String mergeStrategy,
            double confidenceIntervals,
            double predictionIntervals);

    static native NativeResult streamingProcess(long handle, double[] x, double[] y);

    static native NativeResult streamingFinalize(long handle);

    static native void streamingFree(long handle);

    static native long onlineNew(
            double fraction,
            int iterations,
            double delta,
            String weightFunction,
            String robustnessMethod,
            String scalingMethod,
            String boundaryPolicy,
            boolean returnRobustnessWeights,
            boolean returnDiagnostics,
            boolean returnResiduals,
            String zeroWeightFallback,
            double autoConverge,
            boolean parallel,
            int windowCapacity,
            int minPoints,
            String updateMode,
            double confidenceIntervals,
            double predictionIntervals);

    static native NativeOnlineOutput onlineAddPoint(long handle, double x, double y);

    static native void onlineFree(long handle);
}
