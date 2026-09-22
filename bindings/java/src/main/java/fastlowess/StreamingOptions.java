package fastlowess;

/**
 * Configuration for a {@link StreamingLowess} model. Construct via
 * {@link #builder()}.
 */
public final class StreamingOptions {

    final Options common;
    final int chunkSize;
    final int overlap;
    final String mergeStrategy;

    StreamingOptions(Builder b) {
        this.common = b.common.build();
        this.chunkSize = b.chunkSize;
        this.overlap = b.overlap;
        this.mergeStrategy = b.mergeStrategy;
    }

    /**
     * Creates a new builder.
     *
     * @return a new {@link Builder}
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Fluent builder for {@link StreamingOptions}.
     */
    public static final class Builder {

        private final Options.Builder common = Options.builder();
        int chunkSize = 5000;
        int overlap = -1;
        String mergeStrategy = null;

        Builder() {
        }

        /**
         * The fraction of points used to compute each local regression.
         *
         * @param fraction the fraction of points used to compute each local
         * regression
         * @return this builder, for chaining
         * @see Options.Builder#fraction(double)
         */
        public Builder fraction(double fraction) {
            common.fraction(fraction);
            return this;
        }

        /**
         * The number of robustifying iterations.
         *
         * @param iterations the number of robustifying iterations
         * @return this builder, for chaining
         * @see Options.Builder#iterations(int)
         */
        public Builder iterations(int iterations) {
            common.iterations(iterations);
            return this;
        }

        /**
         * Skips recomputation for points within this distance of the last fit
         * point.
         *
         * @param delta the interpolation distance
         * @return this builder, for chaining
         * @see Options.Builder#delta(double)
         */
        public Builder delta(double delta) {
            common.delta(delta);
            return this;
        }

        /**
         * One of
         * {@code "tricube"}, {@code "epanechnikov"}, {@code "gaussian"}, {@code "uniform"}, {@code "biweight"}, {@code "triangle"}, {@code "cosine"}.
         *
         * @param weightFunction the weight function name
         * @return this builder, for chaining
         * @see Options.Builder#weightFunction(String)
         */
        public Builder weightFunction(String weightFunction) {
            common.weightFunction(weightFunction);
            return this;
        }

        /**
         * One of {@code "bisquare"}, {@code "huber"}, {@code "talwar"}.
         *
         * @param robustnessMethod the robustness method name
         * @return this builder, for chaining
         * @see Options.Builder#robustnessMethod(String)
         */
        public Builder robustnessMethod(String robustnessMethod) {
            common.robustnessMethod(robustnessMethod);
            return this;
        }

        /**
         * One of {@code "mad"}, {@code "mar"}, {@code "mean"}.
         *
         * @param scalingMethod the residual scaling method name
         * @return this builder, for chaining
         * @see Options.Builder#scalingMethod(String)
         */
        public Builder scalingMethod(String scalingMethod) {
            common.scalingMethod(scalingMethod);
            return this;
        }

        /**
         * One of
         * {@code "extend"}, {@code "reflect"}, {@code "zero"}, {@code "noboundary"}.
         *
         * @param boundaryPolicy the boundary handling policy name
         * @return this builder, for chaining
         * @see Options.Builder#boundaryPolicy(String)
         */
        public Builder boundaryPolicy(String boundaryPolicy) {
            common.boundaryPolicy(boundaryPolicy);
            return this;
        }

        /**
         * How to handle all-zero local weight windows.
         *
         * @param zeroWeightFallback the zero-weight handling strategy name
         * @return this builder, for chaining
         * @see Options.Builder#zeroWeightFallback(String)
         */
        public Builder zeroWeightFallback(String zeroWeightFallback) {
            common.zeroWeightFallback(zeroWeightFallback);
            return this;
        }

        /**
         * Policy for non-finite (NaN/Infinity) values in input data:
         * {@code "error"} throws, {@code "drop"} silently removes affected
         * observations before fitting.
         *
         * @param missing the missing-value handling policy name
         * @return this builder, for chaining
         * @see Options.Builder#missing(String)
         */
        public Builder missing(String missing) {
            common.missing(missing);
            return this;
        }

        /**
         * Stops iterating early once the relative change in fitted values drops
         * below this value.
         *
         * @param autoConverge the auto-convergence tolerance
         * @return this builder, for chaining
         * @see Options.Builder#autoConverge(double)
         */
        public Builder autoConverge(double autoConverge) {
            common.autoConverge(autoConverge);
            return this;
        }

        /**
         * Whether {@link Result#diagnostics()} should be populated.
         *
         * @param returnDiagnostics whether to compute diagnostics
         * @return this builder, for chaining
         * @see Options.Builder#returnDiagnostics(boolean)
         */
        public Builder returnDiagnostics(boolean returnDiagnostics) {
            common.returnDiagnostics(returnDiagnostics);
            return this;
        }

        /**
         * Whether {@link Result#residuals()} should be populated.
         *
         * @param returnResiduals whether to include residuals in the result
         * @return this builder, for chaining
         * @see Options.Builder#returnResiduals(boolean)
         */
        public Builder returnResiduals(boolean returnResiduals) {
            common.returnResiduals(returnResiduals);
            return this;
        }

        /**
         * Whether {@link Result#robustnessWeights()} should be populated.
         *
         * @param returnRobustnessWeights whether to include robustness weights
         * in the result
         * @return this builder, for chaining
         * @see Options.Builder#returnRobustnessWeights(boolean)
         */
        public Builder returnRobustnessWeights(boolean returnRobustnessWeights) {
            common.returnRobustnessWeights(returnRobustnessWeights);
            return this;
        }

        /**
         * Whether {@link Result#derivative()} should be populated.
         *
         * @param returnDerivative whether to include the per-point local fit
         * derivative (slope) in the result
         * @return this builder, for chaining
         * @see Options.Builder#returnDerivative(boolean)
         */
        public Builder returnDerivative(boolean returnDerivative) {
            common.returnDerivative(returnDerivative);
            return this;
        }

        /**
         * Whether {@link Result#standardErrors()} should be populated.
         *
         * @param returnSe whether to include standard errors in the result
         * @return this builder, for chaining
         * @see Options.Builder#returnSe(boolean)
         */
        public Builder returnSe(boolean returnSe) {
            common.returnSe(returnSe);
            return this;
        }

        /**
         * Requests confidence intervals at the given level (e.g. {@code 0.95}).
         *
         * @param confidenceIntervals the confidence level (e.g. 0.95)
         * @return this builder, for chaining
         * @see Options.Builder#confidenceIntervals(double)
         */
        public Builder confidenceIntervals(double confidenceIntervals) {
            common.confidenceIntervals(confidenceIntervals);
            return this;
        }

        /**
         * Requests prediction intervals at the given level (e.g. {@code 0.95}).
         *
         * @param predictionIntervals the prediction level (e.g. 0.95)
         * @return this builder, for chaining
         * @see Options.Builder#predictionIntervals(double)
         */
        public Builder predictionIntervals(double predictionIntervals) {
            common.predictionIntervals(predictionIntervals);
            return this;
        }

        /**
         * Whether to use the multi-threaded execution path.
         *
         * @param parallel whether to enable parallel execution
         * @return this builder, for chaining
         * @see Options.Builder#parallel(boolean)
         */
        public Builder parallel(boolean parallel) {
            common.parallel(parallel);
            return this;
        }

        /**
         * Number of points processed per chunk (default {@code 5000}).
         *
         * @param chunkSize the chunk size
         * @return this builder, for chaining
         */
        public Builder chunkSize(int chunkSize) {
            this.chunkSize = chunkSize;
            return this;
        }

        /**
         * Number of points overlapped between consecutive chunks (default:
         * library default of {@code chunk_size / 10}, clamped to
         * {@code [1, chunk_size - 10]}). Any negative value means "use the
         * library default".
         *
         * @param overlap the overlap size
         * @return this builder, for chaining
         */
        public Builder overlap(int overlap) {
            this.overlap = overlap;
            return this;
        }

        /**
         * One of
         * {@code "average"}, {@code "weighted_average"}, {@code "take_first"}, {@code "take_last"}
         * (default {@code "weighted_average"}).
         *
         * @param mergeStrategy the merge strategy name
         * @return this builder, for chaining
         */
        public Builder mergeStrategy(String mergeStrategy) {
            this.mergeStrategy = mergeStrategy;
            return this;
        }

        /**
         * Selects optional result components: {@code "diagnostics"},
         * {@code "residuals"}, {@code "weights"}, {@code "derivative"}, and
         * {@code "se"}.
         *
         * @param outputs optional output component names
         * @return this builder, for chaining
         */
        public Builder outputs(String... outputs) {
            for (String output : outputs) {
                switch (output) {
                    case "diagnostics", "residuals", "weights", "derivative", "se" -> {
                        this.common.outputs(output);
                    }
                    default ->
                        throw new IllegalArgumentException("Unknown output: " + output);
                }
            }
            return this;
        }

        /**
         * Builds the immutable {@link StreamingOptions}.
         *
         * @return the constructed options
         */
        public StreamingOptions build() {
            return new StreamingOptions(this);
        }
    }
}
