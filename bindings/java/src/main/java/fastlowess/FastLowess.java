package fastlowess;

/**
 * Static utility methods for the fastlowess native library.
 */
public final class FastLowess {

    private FastLowess() {
    }

    /**
     * Returns the fastLowess crate version backing this binding.
     */
    public static String version() {
        return NativeBridge.version();
    }

    /**
     * Returns true if this library was built with the GPU execution backend
     * enabled.
     */
    public static boolean gpuEnabled() {
        return NativeBridge.gpuEnabled();
    }
}
