package fastlowess;

/**
 * Options for {@link PredictModel#predict}. Construct via {@link #builder()}.
 *
 * @param returnSe whether to include standard errors in the output
 * @param returnDerivative whether to include the local fit's derivative (slope)
 * in the output
 * @param intervals confidence/prediction levels and optional residual-bootstrap
 * refits
 * @param seed prediction-time bootstrap seed, or {@code null} for the default;
 * interpreted as an unsigned 64-bit integer
 * @param extrapolation behavior for query points outside the training range:
 * one of {@code "clamp"} (default), {@code "linear"}, {@code "error"}
 * @param maxExtrapolationDistance under {@code "linear"} extrapolation, the
 * maximum allowed distance beyond the training boundary before {@code predict}
 * throws instead of returning an unbounded value, or {@code Double.NaN} to
 * disable
 * @param maxNeighborDistance maximum allowed distance to the farthest point in
 * a query's local window before {@code predict} throws, catching
 * in-range-but-sparse query points, or {@code Double.NaN} to disable
 */
public record PredictOptions(
        boolean returnSe,
        boolean returnDerivative,
        IntervalsOptions intervals,
        Long seed,
        String extrapolation,
        double maxExtrapolationDistance,
        double maxNeighborDistance) {

    /**
     * Normalizes a {@code null} interval group to "disabled".
     */
    public PredictOptions {
        intervals = intervals == null ? IntervalsOptions.DISABLED : intervals;
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
     * Fluent builder for {@link PredictOptions}.
     */
    public static final class Builder {

        boolean returnSe = false;
        boolean returnDerivative = false;
        IntervalsOptions intervals = IntervalsOptions.DISABLED;
        Long seed = null;
        String extrapolation = "clamp";
        double maxExtrapolationDistance = Double.NaN;
        double maxNeighborDistance = Double.NaN;

        Builder() {
        }

        /**
         * Whether to include standard errors in the output (default
         * {@code false}).
         *
         * @param returnSe whether to include standard errors
         * @return this builder, for chaining
         */
        public Builder returnSe(boolean returnSe) {
            this.returnSe = returnSe;
            return this;
        }

        /**
         * Whether to include the local fit's derivative (slope) in the output
         * (default {@code false}).
         *
         * @param returnDerivative whether to include the derivative
         * @return this builder, for chaining
         */
        public Builder returnDerivative(boolean returnDerivative) {
            this.returnDerivative = returnDerivative;
            return this;
        }

        /**
         * Selects optional prediction components: {@code "se"} and/or
         * {@code "derivative"}.
         *
         * @param outputs optional prediction component names
         * @return this builder, for chaining
         */
        public Builder outputs(String... outputs) {
            for (String output : outputs) {
                switch (output) {
                    case "se" ->
                        this.returnSe = true;
                    case "derivative" ->
                        this.returnDerivative = true;
                    default ->
                        throw new IllegalArgumentException("Unknown output: " + output);
                }

            }
            return this;
        }

        /**
         * Confidence/prediction levels and optional residual-bootstrap refits
         * over the retained training residuals.
         *
         * @param intervals interval configuration
         * @return this builder, for chaining
         */
        public Builder intervals(IntervalsOptions intervals) {
            this.intervals = intervals;
            return this;
        }

        /**
         * Seeds prediction-time bootstrap draws, independently of the fit's
         * seed. Does not enable bootstrap by itself; {@code 0} is valid.
         *
         * @param seed the random seed
         * @return this builder, for chaining
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * Behavior for query points outside the training range: one of
         * {@code "clamp"} (default), {@code "linear"}, {@code "error"}.
         *
         * @param extrapolation the extrapolation policy name
         * @return this builder, for chaining
         */
        public Builder extrapolation(String extrapolation) {
            this.extrapolation = extrapolation;
            return this;
        }

        /**
         * Under {@code "linear"} extrapolation, the maximum allowed distance
         * beyond the training boundary before {@code predict} throws instead of
         * returning an unbounded value.
         *
         * @param maxExtrapolationDistance the maximum extrapolation distance
         * @return this builder, for chaining
         */
        public Builder maxExtrapolationDistance(double maxExtrapolationDistance) {
            this.maxExtrapolationDistance = maxExtrapolationDistance;
            return this;
        }

        /**
         * Maximum allowed distance to the farthest point in a query's local
         * window before {@code predict} throws, catching in-range-but-sparse
         * query points.
         *
         * @param maxNeighborDistance the maximum neighbor distance
         * @return this builder, for chaining
         */
        public Builder maxNeighborDistance(double maxNeighborDistance) {
            this.maxNeighborDistance = maxNeighborDistance;
            return this;
        }

        /**
         * Builds the immutable {@link PredictOptions}.
         *
         * @return the constructed options
         */
        public PredictOptions build() {
            return new PredictOptions(
                    returnSe,
                    returnDerivative,
                    intervals,
                    seed,
                    extrapolation,
                    maxExtrapolationDistance,
                    maxNeighborDistance);
        }
    }
}
