package fastlowess;

/**
 * Retained fitted-model state enabling out-of-sample {@link #predict}, obtained
 * via {@link Result#predictModel()} when
 * {@link Options.Builder#retainModel(boolean)} was set to {@code true} before
 * {@link Lowess#fit}.
 *
 * <p>
 * Operations on one instance are synchronized; each instance wraps a native
 * handle that must be freed via {@link #close()} or {@link Result#close()}. A
 * {@link java.lang.ref.Cleaner} is registered as a safety net.
 */
public final class PredictModel implements AutoCloseable {

    private final NativeHandles.Owned nativeHandle;

    PredictModel(long handle) {
        this.nativeHandle = new NativeHandles.Owned(handle, NativeBridge::predictHandleFree);
    }

    /**
     * Evaluates the fitted model at out-of-sample query points not in the
     * training set, using default options.
     *
     * @param newX the query points
     * @return the prediction result
     */
    public synchronized PredictResult predict(double[] newX) {
        return predict(newX, PredictOptions.builder().build());
    }

    /**
     * Evaluates the fitted model at out-of-sample query points not in the
     * training set.
     *
     * @param newX the query points
     * @param options prediction options
     * @return the prediction result
     */
    public synchronized PredictResult predict(double[] newX, PredictOptions options) {
        checkOpen();
        Inputs.requireQueryX(newX);
        IntervalsOptions iv = options.intervals();
        Long seed = options.seed();
        NativePredictResult r = NativeBridge.predict(
                nativeHandle.get(),
                newX,
                options.outputs().contains("se"),
                iv.confidence,
                iv.prediction,
                options.outputs().contains("derivative"),
                options.extrapolation(),
                options.maxExtrapolationDistance(),
                options.maxNeighborDistance(),
                iv.bootstrap,
                seed == null ? 0L : seed,
                seed != null);
        return PredictResult.fromNative(r);
    }

    private void checkOpen() {
        if (!nativeHandle.isOpen()) {
            throw new IllegalStateException("PredictModel has already been closed");
        }
    }

    @Override
    public synchronized void close() {
        nativeHandle.close();
    }
}
