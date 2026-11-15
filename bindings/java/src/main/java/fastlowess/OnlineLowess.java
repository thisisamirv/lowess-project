package fastlowess;

import java.util.Optional;

/**
 * An online LOWESS model that updates incrementally as points arrive.
 * Operations on one instance are synchronized because it owns mutable native
 * state.
 */
public final class OnlineLowess implements AutoCloseable {

    private final NativeHandles.Owned nativeHandle;

    /**
     * Creates a new online model from the given options.
     *
     * @param options the model configuration
     */
    public OnlineLowess(OnlineOptions options) {
        Options c = options.common;
        Long seed = c.seed;
        long handle = NativeBridge.onlineNew(
                c.fraction,
                c.iterations,
                c.delta,
                c.weightFunction,
                c.robustnessMethod,
                c.scalingMethod,
                c.boundaryPolicy,
                c.returnRobustnessWeights,
                c.returnDerivative,
                c.zeroWeightFallback,
                c.autoConverge,
                options.windowCapacity,
                options.minPoints,
                options.updateMode,
                c.missing,
                c.returnSe,
                c.intervals.confidence,
                c.intervals.prediction,
                c.intervals.bootstrap,
                seed == null ? 0L : seed,
                seed != null);
        this.nativeHandle = new NativeHandles.Owned(handle, NativeBridge::onlineFree);
    }

    /**
     * Adds a point to the model.
     *
     * <p>
     * {@link OnlinePointStatus#FILLING} while fewer than {@code minPoints}
     * finite observations have been accepted;
     * {@link OnlinePointStatus#READY} once smoothing begins (see
     * {@link AddPointResult#point()});
     * {@link OnlinePointStatus#DROPPED} when {@code missing = "drop"} ignores a
     * non-finite {@code x} or {@code y}.
     *
     * @param x the x value
     * @param y the y value
     * @return status and optional smoothed output
     */
    public synchronized AddPointResult addPoint(double x, double y) {
        checkOpen();
        NativeOnlineOutput o = NativeBridge.onlineAddPoint(nativeHandle.get(), x, y);
        OnlinePointStatus status = OnlinePointStatus.fromNative(o.hasValue);
        return new AddPointResult(
                status,
                status.isReady() ? Optional.of(PointResult.fromNative(o)) : Optional.empty());
    }

    private void checkOpen() {
        if (!nativeHandle.isOpen()) {
            throw new IllegalStateException("OnlineLowess has already been closed");
        }
    }

    @Override
    public synchronized void close() {
        nativeHandle.close();
    }
}
