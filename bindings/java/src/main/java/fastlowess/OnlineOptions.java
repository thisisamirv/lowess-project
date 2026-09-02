package fastlowess;

/**
 * Configuration for an {@link OnlineLowess} model. Construct via
 * {@link #builder()}.
 */
public final class OnlineOptions {

    final Options common;
    final int windowCapacity;
    final int minPoints;
    final String updateMode;

    OnlineOptions(Builder b) {
        this.common = b.common.build();
        this.windowCapacity = b.windowCapacity;
        this.minPoints = b.minPoints;
        this.updateMode = b.updateMode;
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
     * Fluent builder for {@link OnlineOptions}.
     */
    public static final class Builder {

        private final Options.Builder common = Options.builder();
        int windowCapacity = 1000;
        int minPoints = 2;
        String updateMode = null;

        Builder() {
        }

        /**
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
         * @param iterations the number of robustifying iterations
         * @return this builder, for chaining
         * @see Options.Builder#iterations(int)
         */
        public Builder iterations(int iterations) {
            common.iterations(iterations);
            return this;
        }

        /**
         * @param delta the interpolation distance
         * @return this builder, for chaining
         * @see Options.Builder#delta(double)
         */
        public Builder delta(double delta) {
            common.delta(delta);
            return this;
        }

        /**
         * @param weightFunction the weight function name
         * @return this builder, for chaining
         * @see Options.Builder#weightFunction(String)
         */
        public Builder weightFunction(String weightFunction) {
            common.weightFunction(weightFunction);
            return this;
        }

        /**
         * @param robustnessMethod the robustness method name
         * @return this builder, for chaining
         * @see Options.Builder#robustnessMethod(String)
         */
        public Builder robustnessMethod(String robustnessMethod) {
            common.robustnessMethod(robustnessMethod);
            return this;
        }

        /**
         * @param scalingMethod the residual scaling method name
         * @return this builder, for chaining
         * @see Options.Builder#scalingMethod(String)
         */
        public Builder scalingMethod(String scalingMethod) {
            common.scalingMethod(scalingMethod);
            return this;
        }

        /**
         * @param boundaryPolicy the boundary handling policy name
         * @return this builder, for chaining
         * @see Options.Builder#boundaryPolicy(String)
         */
        public Builder boundaryPolicy(String boundaryPolicy) {
            common.boundaryPolicy(boundaryPolicy);
            return this;
        }

        /**
         * @param zeroWeightFallback the zero-weight handling strategy name
         * @return this builder, for chaining
         * @see Options.Builder#zeroWeightFallback(String)
         */
        public Builder zeroWeightFallback(String zeroWeightFallback) {
            common.zeroWeightFallback(zeroWeightFallback);
            return this;
        }

        /**
         * @param autoConverge the auto-convergence tolerance
         * @return this builder, for chaining
         * @see Options.Builder#autoConverge(double)
         */
        public Builder autoConverge(double autoConverge) {
            common.autoConverge(autoConverge);
            return this;
        }

        /**
         * @param confidenceIntervals the confidence level
         * @return this builder, for chaining
         * @see Options.Builder#confidenceIntervals(double)
         */
        public Builder confidenceIntervals(double confidenceIntervals) {
            common.confidenceIntervals(confidenceIntervals);
            return this;
        }

        /**
         * @param predictionIntervals the prediction level
         * @return this builder, for chaining
         * @see Options.Builder#predictionIntervals(double)
         */
        public Builder predictionIntervals(double predictionIntervals) {
            common.predictionIntervals(predictionIntervals);
            return this;
        }

        /**
         * @param returnDiagnostics whether to compute diagnostics
         * @return this builder, for chaining
         * @see Options.Builder#returnDiagnostics(boolean)
         */
        public Builder returnDiagnostics(boolean returnDiagnostics) {
            common.returnDiagnostics(returnDiagnostics);
            return this;
        }

        /**
         * @param returnResiduals whether to include residuals in the result
         * @return this builder, for chaining
         * @see Options.Builder#returnResiduals(boolean)
         */
        public Builder returnResiduals(boolean returnResiduals) {
            common.returnResiduals(returnResiduals);
            return this;
        }

        /**
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
         * @param parallel whether to enable parallel execution
         * @return this builder, for chaining
         * @see Options.Builder#parallel(boolean)
         */
        public Builder parallel(boolean parallel) {
            common.parallel(parallel);
            return this;
        }

        /**
         * Maximum number of points retained in the sliding window (default
         * {@code 1000}).
         *
         * @param windowCapacity the maximum window size
         * @return this builder, for chaining
         */
        public Builder windowCapacity(int windowCapacity) {
            this.windowCapacity = windowCapacity;
            return this;
        }

        /**
         * Minimum number of points required before a fit is produced (default
         * {@code 2}).
         *
         * @param minPoints the minimum point count
         * @return this builder, for chaining
         */
        public Builder minPoints(int minPoints) {
            this.minPoints = minPoints;
            return this;
        }

        /**
         * One of {@code "incremental"}, {@code "full"} (default
         * {@code "incremental"}).
         *
         * @param updateMode the update mode name
         * @return this builder, for chaining
         */
        public Builder updateMode(String updateMode) {
            this.updateMode = updateMode;
            return this;
        }

        /**
         * Builds the immutable {@link OnlineOptions}.
         *
         * @return the constructed options
         */
        public OnlineOptions build() {
            return new OnlineOptions(this);
        }
    }
}
