package fastlowess;

/**
 * A streaming LOWESS model that processes data in chunks.
 */
public final class StreamingLowess implements AutoCloseable {

    private long handle;

    /**
     * Creates a new streaming model from the given options.
     *
     * @param options the model configuration
     */
    public StreamingLowess(StreamingOptions options) {
        Options c = options.common;
        this.handle = NativeBridge.streamingNew(
                c.fraction,
                c.iterations,
                c.delta,
                c.weightFunction,
                c.robustnessMethod,
                c.scalingMethod,
                c.boundaryPolicy,
                c.returnDiagnostics,
                c.returnResiduals,
                c.returnRobustnessWeights,
                c.zeroWeightFallback,
                c.autoConverge,
                c.parallel,
                options.chunkSize,
                options.overlap,
                options.mergeStrategy,
                c.confidenceIntervals,
                c.predictionIntervals);
    }

    /**
     * Processes the next chunk of data, returning a partial fit for it.
     *
     * @param x the x values of this chunk
     * @param y the y values of this chunk
     * @return the partial fit result for this chunk
     */
    public Result processChunk(double[] x, double[] y) {
        checkOpen();
        NativeResult r = NativeBridge.streamingProcess(handle, x, y);
        return Result.fromNative(r);
    }

    /**
     * Merges all processed chunks into a final result.
     *
     * @return the final merged result
     */
    public Result finish() {
        checkOpen();
        NativeResult r = NativeBridge.streamingFinalize(handle);
        return Result.fromNative(r);
    }

    private void checkOpen() {
        if (handle == 0) {
            throw new IllegalStateException("StreamingLowess has already been closed");
        }
    }

    @Override
    public void close() {
        if (handle != 0) {
            NativeBridge.streamingFree(handle);
            handle = 0;
        }
    }
}
