package fastlowess;

import java.util.Optional;

/**
 * Result of {@link OnlineLowess#addPoint(double, double)}.
 *
 * @param status whether the window is still filling, a smoothed point is ready,
 * or the input was dropped under {@code missing = "drop"}
 * @param point the smoothed output when {@code status} is
 * {@link OnlinePointStatus#READY}; empty otherwise
 */
public record AddPointResult(OnlinePointStatus status, Optional<PointResult> point) {

    /**
     * Whether this result includes a smoothed point.
     *
     * @return {@code true} when {@link #status()} is
     * {@link OnlinePointStatus#READY}
     */
    public boolean isReady() {
        return status.isReady();
    }
}
