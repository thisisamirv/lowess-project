package fastlowess;

/**
 * Grouped cross-validation configuration for {@link Options.Builder#cv}.
 * Construct via {@link #builder()}.
 */
public final class CVOptions {

    final double[] fractions;
    final String method;
    final int k;
    final Long seed;

    private CVOptions(Builder builder) {
        this.fractions = builder.fractions.clone();
        this.method = builder.method;
        this.k = builder.k;
        this.seed = builder.seed;
    }

    /**
     * Creates a new cross-validation builder.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Fluent builder for {@link CVOptions}.
     */
    public static final class Builder {

        double[] fractions;
        String method = "kfold";
        int k = 5;
        Long seed;

        Builder() {
        }

        /**
         * Sets candidate smoothing fractions.
         *
         * @param fractions candidate smoothing fractions
         * @return this builder, for chaining
         */
        public Builder fractions(double... fractions) {
            this.fractions = fractions.clone();
            return this;
        }

        /**
         * Sets the cross-validation method: {@code "kfold"} or {@code "loocv"}.
         *
         * @param method cross-validation method
         * @return this builder, for chaining
         */
        public Builder method(String method) {
            this.method = method;
            return this;
        }

        /**
         * Sets the number of k-fold splits. Ignored for LOOCV.
         *
         * @param k number of folds
         * @return this builder, for chaining
         */
        public Builder k(int k) {
            this.k = k;
            return this;
        }

        /**
         * Sets the reproducible k-fold seed. Ignored for LOOCV.
         *
         * @param seed random seed
         * @return this builder, for chaining
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * Builds the immutable options.
         *
         * @return the constructed cross-validation options
         */
        public CVOptions build() {
            if (fractions == null) {
                throw new IllegalStateException("CV fractions must be provided");
            }
            return new CVOptions(this);
        }
    }
}
