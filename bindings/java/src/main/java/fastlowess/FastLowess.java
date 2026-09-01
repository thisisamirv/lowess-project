package fastlowess;

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
    public static final String VERSION = "3.2.0";

    /**
     * Returns the version of this Java binding.
     */
    public static String version() {
        return VERSION;
    }

    /**
     * Returns true if this library was built with the GPU execution backend
     * enabled.
     */
    public static boolean gpuEnabled() {
        return NativeBridge.gpuEnabled();
    }
}
