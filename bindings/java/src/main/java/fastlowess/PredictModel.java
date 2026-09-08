package fastlowess;

/**
 * Retained fitted-model state enabling out-of-sample {@link #predict}, obtained
 * via {@link Result#predictModel()} when
 * {@link Options.Builder#retainModel(boolean)} was set to {@code true} before
 * {@link Lowess#fit}.
 *
 * <p>
 * Not thread-safe; each instance wraps a native handle that must be freed.
 */
public final class PredictModel implements AutoCloseable {

    private long handle;

    PredictModel(long handle) {
        this.handle = handle;
    }

    /**
     * Evaluates the fitted model at out-of-sample query points not in the
     * training set, using default options.
     *
     * @param newX the query points
     * @return the prediction result
     */
    public PredictResult predict(double[] newX) {
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
    public PredictResult predict(double[] newX, PredictOptions options) {
        checkOpen();
        NativePredictResult r = NativeBridge.predict(
                handle,
                newX,
                options.returnSe(),
                options.confidenceLevel(),
                options.predictionLevel(),
                options.returnDerivative(),
                options.extrapolation(),
                options.maxExtrapolationDistance(),
                options.maxNeighborDistance());
        return PredictResult.fromNative(r);
    }

    private void checkOpen() {
        if (handle == 0) {
            throw new IllegalStateException("PredictModel has already been closed");
        }
    }

    @Override
    public void close() {
        if (handle != 0) {
            NativeBridge.predictHandleFree(handle);
            handle = 0;
        }
    }
}
