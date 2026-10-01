package fastlowess;

/**
 * Grouped confidence/prediction interval levels and optional residual-bootstrap
 * refits, shared by every adapter and by {@link PredictOptions}. Construct via
 * {@link #builder()}.
 */
public final class IntervalsOptions {

    final double confidence;
    final double prediction;
    final int bootstrap;

    private IntervalsOptions(Builder builder) {
        this.confidence = builder.confidence;
        this.prediction = builder.prediction;
        this.bootstrap = builder.bootstrap;
    }

    static final IntervalsOptions DISABLED = builder().build();

    /**
     * Creates a new intervals builder.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Fluent builder for {@link IntervalsOptions}.
     */
    public static final class Builder {

        double confidence = Double.NaN;
        double prediction = Double.NaN;
        int bootstrap = 0;

        Builder() {
        }

        /**
         * Requests confidence intervals for the mean response at the given
         * level (e.g. {@code 0.95}).
         *
         * @param level the confidence level
         * @return this builder, for chaining
         */
        public Builder confidence(double level) {
            this.confidence = level;
            return this;
        }

        /**
         * Requests prediction intervals for a new observation at the given
         * level (e.g. {@code 0.95}).
         *
         * @param level the prediction level
         * @return this builder, for chaining
         */
        public Builder prediction(double level) {
            this.prediction = level;
            return this;
        }

        /**
         * Replaces analytic standard errors and intervals with this many
         * residual-bootstrap refits (at least {@code 2}). {@code 0} (default)
         * keeps the analytic intervals.
         *
         * @param replicates the number of bootstrap refits
         * @return this builder, for chaining
         */
        public Builder bootstrap(int replicates) {
            this.bootstrap = replicates;
            return this;
        }

        /**
         * Builds the immutable options.
         *
         * @return the constructed interval options
         */
        public IntervalsOptions build() {
            return new IntervalsOptions(this);
        }
    }
}
