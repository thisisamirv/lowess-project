package fastlowess;

/**
 * A batch LOWESS model. Not thread-safe; each instance wraps a native handle
 * that must be freed.
 */
public final class Lowess implements AutoCloseable {

    private long handle;

    /**
     * Creates a new batch model from the given options.
     *
     * @param options the model configuration
     */
    public Lowess(Options options) {
        IntervalsOptions iv = options.intervals;
        CVOptions cv = options.cv;
        Long seed = options.seed;
        this.handle = NativeBridge.lowessNew(
                options.fraction,
                options.iterations,
                options.delta,
                options.weightFunction,
                options.robustnessMethod,
                options.scalingMethod,
                options.boundaryPolicy,
                iv.confidence,
                iv.prediction,
                iv.bootstrap,
                options.returnDiagnostics,
                options.returnResiduals,
                options.returnRobustnessWeights,
                options.returnDerivative,
                options.zeroWeightFallback,
                options.autoConverge,
                cv == null ? null : cv.fractions,
                cv == null ? null : cv.method,
                cv == null ? 5 : cv.k,
                seed == null ? 0L : seed,
                seed != null,
                options.parallel,
                options.returnSe,
                options.returnSorted,
                options.backend,
                options.missing,
                options.retainModel);
    }

    /**
     * Fits the model to {@code x}/{@code y}, using uniform weights.
     *
     * @param x the x values
     * @param y the y values
     * @return the fit result
     */
    public Result fit(double[] x, double[] y) {
        return fit(x, y, null);
    }

    /**
     * Fits the model to {@code x}/{@code y}, using the given per-point weights.
     *
     * @param x the x values
     * @param y the y values
     * @param customWeights non-negative per-observation weights, or
     * {@code null}
     * @return the fit result
     */
    public Result fit(double[] x, double[] y, double[] customWeights) {
        checkOpen();
        NativeResult r = NativeBridge.lowessFit(handle, x, y, customWeights);
        return Result.fromNative(r);
    }

    private void checkOpen() {
        if (handle == 0) {
            throw new IllegalStateException("Lowess has already been closed");
        }
    }

    @Override
    public void close() {
        if (handle != 0) {
            NativeBridge.lowessFree(handle);
            handle = 0;
        }
    }
}
