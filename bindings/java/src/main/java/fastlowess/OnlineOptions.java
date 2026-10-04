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

        {
            // Online's default "incremental" mode performs a non-robust
            // single-point fit; robustness iterations require updateMode("full").
            common.iterations(0);
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
         * Confidence/prediction intervals and optional residual-bootstrap
         * refits for each sliding window. Requires {@code updateMode("full")}.
         *
         * @param intervals interval configuration
         * @return this builder, for chaining
         * @see Options.Builder#intervals(IntervalsOptions)
         */
        public Builder intervals(IntervalsOptions intervals) {
            common.intervals(intervals);
            return this;
        }

        /**
         * Seeds bootstrap draws; each full-update window restarts from this
         * seed.
         *
         * @param seed the random seed
         * @return this builder, for chaining
         * @see Options.Builder#seed(long)
         */
        public Builder seed(long seed) {
            common.seed(seed);
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
         * Selects optional result components: {@code "weights"},
         * {@code "derivative"}, and {@code "se"}.
         *
         * @param outputs optional output component names
         * @return this builder, for chaining
         */
        public Builder outputs(String... outputs) {
            for (String output : outputs) {
                switch (output) {
                    case "weights", "derivative", "se" ->
                        this.common.outputs(output);
                    default ->
                        throw new IllegalArgumentException("Unknown output: " + output);
                }
            }
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
