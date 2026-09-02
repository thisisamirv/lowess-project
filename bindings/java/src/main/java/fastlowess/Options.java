package fastlowess;

/**
 * Configuration for a batch {@link Lowess} fit. Construct via
 * {@link #builder()}.
 *
 * <p>
 * {@link StreamingOptions} and {@link OnlineOptions} hold their own instance of
 * the shared subset of these settings (composition), plus settings specific to
 * their execution mode.
 */
public final class Options {

    final double fraction;
    final int iterations;
    final double delta;
    final String weightFunction;
    final String robustnessMethod;
    final String scalingMethod;
    final String boundaryPolicy;
    final String zeroWeightFallback;
    final double autoConverge;
    final double confidenceIntervals;
    final double predictionIntervals;
    final boolean returnDiagnostics;
    final boolean returnResiduals;
    final boolean returnRobustnessWeights;
    final boolean parallel;
    final boolean returnSe;
    final String backend;
    final double[] cvFractions;
    final String cvMethod;
    final int cvK;
    final Long cvSeed;

    Options(Builder b) {
        this.fraction = b.fraction;
        this.iterations = b.iterations;
        this.delta = b.delta;
        this.weightFunction = b.weightFunction;
        this.robustnessMethod = b.robustnessMethod;
        this.scalingMethod = b.scalingMethod;
        this.boundaryPolicy = b.boundaryPolicy;
        this.zeroWeightFallback = b.zeroWeightFallback;
        this.autoConverge = b.autoConverge;
        this.confidenceIntervals = b.confidenceIntervals;
        this.predictionIntervals = b.predictionIntervals;
        this.returnDiagnostics = b.returnDiagnostics;
        this.returnResiduals = b.returnResiduals;
        this.returnRobustnessWeights = b.returnRobustnessWeights;
        this.parallel = b.parallel;
        this.returnSe = b.returnSe;
        this.backend = b.backend;
        this.cvFractions = b.cvFractions;
        this.cvMethod = b.cvMethod;
        this.cvK = b.cvK;
        this.cvSeed = b.cvSeed;
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
     * Fluent builder for {@link Options}.
     */
    public static final class Builder {

        double fraction = 0.67;
        int iterations = 3;
        double delta = Double.NaN;
        String weightFunction = null;
        String robustnessMethod = null;
        String scalingMethod = null;
        String boundaryPolicy = null;
        String zeroWeightFallback = null;
        double autoConverge = Double.NaN;
        double confidenceIntervals = Double.NaN;
        double predictionIntervals = Double.NaN;
        boolean returnDiagnostics = false;
        boolean returnResiduals = false;
        boolean returnRobustnessWeights = false;
        boolean parallel = true;
        boolean returnSe = false;
        String backend = null;
        double[] cvFractions = null;
        String cvMethod = null;
        int cvK = 5;
        Long cvSeed = null;

        Builder() {
        }

        /**
         * The fraction of points used to compute each local regression (default
         * {@code 0.67}).
         *
         * @param fraction the fraction of points used to compute each local
         * regression
         * @return this builder, for chaining
         */
        public Builder fraction(double fraction) {
            this.fraction = fraction;
            return this;
        }

        /**
         * The number of robustifying iterations (default {@code 3}).
         *
         * @param iterations the number of robustifying iterations
         * @return this builder, for chaining
         */
        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }

        /**
         * Skips recomputation for points within this distance of the last fit
         * point (default {@code NaN}, which auto-sets it to 1% of the x-range
         * in Batch mode, or {@code 0.0} in Streaming/Online mode).
         *
         * @param delta the interpolation distance
         * @return this builder, for chaining
         */
        public Builder delta(double delta) {
            this.delta = delta;
            return this;
        }

        /**
         * One of
         * {@code "tricube"}, {@code "epanechnikov"}, {@code "gaussian"}, {@code "uniform"}, {@code "biweight"}, {@code "triangle"}, {@code "cosine"}
         * (default {@code "tricube"}).
         *
         * @param weightFunction the weight function name
         * @return this builder, for chaining
         */
        public Builder weightFunction(String weightFunction) {
            this.weightFunction = weightFunction;
            return this;
        }

        /**
         * One of {@code "bisquare"}, {@code "huber"}, {@code "talwar"} (default
         * {@code "bisquare"}).
         *
         * @param robustnessMethod the robustness method name
         * @return this builder, for chaining
         */
        public Builder robustnessMethod(String robustnessMethod) {
            this.robustnessMethod = robustnessMethod;
            return this;
        }

        /**
         * One of {@code "mad"}, {@code "mar"}, {@code "mean"} (default
         * {@code "mad"}).
         *
         * @param scalingMethod the residual scaling method name
         * @return this builder, for chaining
         */
        public Builder scalingMethod(String scalingMethod) {
            this.scalingMethod = scalingMethod;
            return this;
        }

        /**
         * One of
         * {@code "extend"}, {@code "reflect"}, {@code "zero"}, {@code "noboundary"}
         * (default {@code "extend"}).
         *
         * @param boundaryPolicy the boundary handling policy name
         * @return this builder, for chaining
         */
        public Builder boundaryPolicy(String boundaryPolicy) {
            this.boundaryPolicy = boundaryPolicy;
            return this;
        }

        /**
         * How to handle all-zero local weight windows (default
         * {@code "use_local_mean"}).
         *
         * @param zeroWeightFallback the zero-weight handling strategy name
         * @return this builder, for chaining
         */
        public Builder zeroWeightFallback(String zeroWeightFallback) {
            this.zeroWeightFallback = zeroWeightFallback;
            return this;
        }

        /**
         * Stops iterating early once the relative change in fitted values drops
         * below this value.
         *
         * @param autoConverge the auto-convergence tolerance
         * @return this builder, for chaining
         */
        public Builder autoConverge(double autoConverge) {
            this.autoConverge = autoConverge;
            return this;
        }

        /**
         * Requests confidence intervals at the given level (e.g. {@code 0.95}).
         *
         * @param confidenceIntervals the confidence level
         * @return this builder, for chaining
         */
        public Builder confidenceIntervals(double confidenceIntervals) {
            this.confidenceIntervals = confidenceIntervals;
            return this;
        }

        /**
         * Requests prediction intervals at the given level (e.g. {@code 0.95}).
         *
         * @param predictionIntervals the prediction level
         * @return this builder, for chaining
         */
        public Builder predictionIntervals(double predictionIntervals) {
            this.predictionIntervals = predictionIntervals;
            return this;
        }

        /**
         * Whether {@link Result#diagnostics()} should be populated.
         *
         * @param returnDiagnostics whether to compute diagnostics
         * @return this builder, for chaining
         */
        public Builder returnDiagnostics(boolean returnDiagnostics) {
            this.returnDiagnostics = returnDiagnostics;
            return this;
        }

        /**
         * Whether {@link Result#residuals()} should be populated.
         *
         * @param returnResiduals whether to include residuals in the result
         * @return this builder, for chaining
         */
        public Builder returnResiduals(boolean returnResiduals) {
            this.returnResiduals = returnResiduals;
            return this;
        }

        /**
         * Whether {@link Result#robustnessWeights()} should be populated.
         *
         * @param returnRobustnessWeights whether to include robustness weights
         * in the result
         * @return this builder, for chaining
         */
        public Builder returnRobustnessWeights(boolean returnRobustnessWeights) {
            this.returnRobustnessWeights = returnRobustnessWeights;
            return this;
        }

        /**
         * Whether to use the multi-threaded execution path (default
         * {@code true}).
         *
         * @param parallel whether to enable parallel execution
         * @return this builder, for chaining
         */
        public Builder parallel(boolean parallel) {
            this.parallel = parallel;
            return this;
        }

        /**
         * Whether {@link Result#standardErrors()} should be populated.
         *
         * @param returnSe whether to return standard errors
         * @return this builder, for chaining
         */
        public Builder returnSe(boolean returnSe) {
            this.returnSe = returnSe;
            return this;
        }

        /**
         * One of {@code "cpu"}, {@code "gpu"} (default {@code "cpu"}).
         *
         * @param backend the execution backend name
         * @return this builder, for chaining
         */
        public Builder backend(String backend) {
            this.backend = backend;
            return this;
        }

        /**
         * Candidate fractions to cross-validate; enables
         * {@link Result#cvScores()}.
         *
         * @param cvFractions the fractions to test for cross-validation
         * @return this builder, for chaining
         */
        public Builder cvFractions(double[] cvFractions) {
            this.cvFractions = cvFractions;
            return this;
        }

        /**
         * One of {@code "kfold"}, {@code "loocv"} (default {@code "kfold"});
         * only used when {@code cvFractions} is set.
         *
         * @param cvMethod the cross-validation method name
         * @return this builder, for chaining
         */
        public Builder cvMethod(String cvMethod) {
            this.cvMethod = cvMethod;
            return this;
        }

        /**
         * Number of folds for {@code "kfold"} cross-validation (default
         * {@code 5}).
         *
         * @param cvK the number of folds
         * @return this builder, for chaining
         */
        public Builder cvK(int cvK) {
            this.cvK = cvK;
            return this;
        }

        /**
         * Seeds the cross-validation fold assignment for reproducibility.
         *
         * @param cvSeed the random seed
         * @return this builder, for chaining
         */
        public Builder cvSeed(long cvSeed) {
            this.cvSeed = cvSeed;
            return this;
        }

        /**
         * Builds the immutable {@link Options}.
         *
         * @return the constructed options
         */
        public Options build() {
            return new Options(this);
        }
    }
}
