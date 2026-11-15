package fastlowess;

/**
 * Outcome of {@link OnlineLowess#addPoint(double, double)} when no smoothed
 * value is produced, or when one is.
 */
public enum OnlinePointStatus {
    /**
     * Fewer than {@code minPoints} finite observations have been accepted so
     * far; no smoothed output is available yet.
     */
    FILLING,
    /**
     * The associated {@link PointResult} is valid for the latest point.
     */
    READY,
    /**
     * The point was ignored under {@code missing = "drop"} (non-finite
     * {@code x} or {@code y}).
     */
    DROPPED;

    /**
     * @return {@code true} when this status is {@link #READY}
     */
    public boolean isReady() {
        return this == READY;
    }

    static OnlinePointStatus fromNative(int hasValue) {
        if (hasValue > 0) {
            return READY;
        }
        if (hasValue < 0) {
            return DROPPED;
        }
        return FILLING;
    }
}
