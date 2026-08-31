package com.thisisamirv.fastlowess;

/**
 * A batch LOWESS model. Not thread-safe; each instance wraps a native handle
 * that must be freed.
 */
public final class Lowess implements AutoCloseable {

    private long handle;

    public Lowess(Options options) {
        this.handle = NativeBridge.lowessNew(
                options.fraction,
                options.iterations,
                options.delta,
                options.weightFunction,
                options.robustnessMethod,
                options.scalingMethod,
                options.boundaryPolicy,
                options.confidenceIntervals,
                options.predictionIntervals,
                options.returnDiagnostics,
                options.returnResiduals,
                options.returnRobustnessWeights,
                options.zeroWeightFallback,
                options.autoConverge,
                options.cvFractions,
                options.cvMethod,
                options.cvK,
                options.parallel,
                options.returnSe,
                options.backend);
        if (options.cvSeed != null) {
            NativeBridge.lowessSetCvSeed(handle, options.cvSeed);
        }
    }

    /**
     * Fits the model to {@code x}/{@code y}, using uniform weights.
     */
    public Result fit(double[] x, double[] y) {
        return fit(x, y, null);
    }

    /**
     * Fits the model to {@code x}/{@code y}, using the given per-point weights.
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
