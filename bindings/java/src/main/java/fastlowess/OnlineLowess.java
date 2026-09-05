package fastlowess;

import java.util.Optional;

/**
 * An online LOWESS model that updates incrementally as points arrive.
 */
public final class OnlineLowess implements AutoCloseable {

    private long handle;

    /**
     * Creates a new online model from the given options.
     *
     * @param options the model configuration
     */
    public OnlineLowess(OnlineOptions options) {
        Options c = options.common;
        this.handle = NativeBridge.onlineNew(
                c.fraction,
                c.iterations,
                c.delta,
                c.weightFunction,
                c.robustnessMethod,
                c.scalingMethod,
                c.boundaryPolicy,
                c.returnRobustnessWeights,
                c.zeroWeightFallback,
                c.autoConverge,
                options.windowCapacity,
                options.minPoints,
                options.updateMode,
                c.missing);
    }

    /**
     * Adds a point to the model, returning a smoothed output once at least
     * {@code minPoints} have been seen, or {@link Optional#empty()} otherwise.
     *
     * @param x the x value
     * @param y the y value
     * @return the smoothed output, or {@link Optional#empty()} if not enough
     * points have been seen yet
     */
    public Optional<PointResult> addPoint(double x, double y) {
        checkOpen();
        NativeOnlineOutput o = NativeBridge.onlineAddPoint(handle, x, y);
        return o.hasValue ? Optional.of(PointResult.fromNative(o)) : Optional.empty();
    }

    private void checkOpen() {
        if (handle == 0) {
            throw new IllegalStateException("OnlineLowess has already been closed");
        }
    }

    @Override
    public void close() {
        if (handle != 0) {
            NativeBridge.onlineFree(handle);
            handle = 0;
        }
    }
}
