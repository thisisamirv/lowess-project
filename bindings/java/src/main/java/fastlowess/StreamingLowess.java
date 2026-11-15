package fastlowess;

/**
 * A streaming LOWESS model that processes data in chunks. Operations on one
 * instance are synchronized because it owns mutable native state.
 */
public final class StreamingLowess implements AutoCloseable {

    private final NativeHandles.Owned nativeHandle;

    /**
     * Creates a new streaming model from the given options.
     *
     * @param options the model configuration
     */
    public StreamingLowess(StreamingOptions options) {
        Options c = options.common;
        Long seed = c.seed;
        long handle = NativeBridge.streamingNew(
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
                c.returnDerivative,
                c.zeroWeightFallback,
                c.autoConverge,
                c.parallel,
                options.chunkSize,
                options.overlap,
                options.mergeStrategy,
                c.missing,
                c.returnSe,
                c.intervals.confidence,
                c.intervals.prediction,
                c.intervals.bootstrap,
                seed == null ? 0L : seed,
                seed != null);
        this.nativeHandle = new NativeHandles.Owned(handle, NativeBridge::streamingFree);
    }

    /**
     * Processes the next chunk of data, returning a partial fit for it.
     *
     * @param x the x values of this chunk
     * @param y the y values of this chunk
     * @return the partial fit result for this chunk
     */
    public synchronized Result processChunk(double[] x, double[] y) {
        checkOpen();
        Inputs.requireXy(x, y);
        NativeResult r = NativeBridge.streamingProcess(nativeHandle.get(), x, y);
        return Result.fromNative(r);
    }

    /**
     * Merges all processed chunks into a final result.
     *
     * @return the final merged result
     */
    public synchronized Result finish() {
        checkOpen();
        NativeResult r = NativeBridge.streamingFinalize(nativeHandle.get());
        return Result.fromNative(r);
    }

    private void checkOpen() {
        if (!nativeHandle.isOpen()) {
            throw new IllegalStateException("StreamingLowess has already been closed");
        }
    }

    @Override
    public synchronized void close() {
        nativeHandle.close();
    }
}
