package com.thisisamirv.fastlowess;

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

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Fluent builder for {@link StreamingOptions}.
     */
    public static final class Builder {

        private final Options.Builder common = Options.builder();
        int chunkSize = 1000;
        int overlap = 0;
        String mergeStrategy = null;

        Builder() {
        }

        public Builder fraction(double fraction) {
            common.fraction(fraction);
            return this;
        }

        public Builder iterations(int iterations) {
            common.iterations(iterations);
            return this;
        }

        public Builder delta(double delta) {
            common.delta(delta);
            return this;
        }

        public Builder weightFunction(String weightFunction) {
            common.weightFunction(weightFunction);
            return this;
        }

        public Builder robustnessMethod(String robustnessMethod) {
            common.robustnessMethod(robustnessMethod);
            return this;
        }

        public Builder scalingMethod(String scalingMethod) {
            common.scalingMethod(scalingMethod);
            return this;
        }

        public Builder boundaryPolicy(String boundaryPolicy) {
            common.boundaryPolicy(boundaryPolicy);
            return this;
        }

        public Builder zeroWeightFallback(String zeroWeightFallback) {
            common.zeroWeightFallback(zeroWeightFallback);
            return this;
        }

        public Builder autoConverge(double autoConverge) {
            common.autoConverge(autoConverge);
            return this;
        }

        public Builder confidenceIntervals(double confidenceIntervals) {
            common.confidenceIntervals(confidenceIntervals);
            return this;
        }

        public Builder predictionIntervals(double predictionIntervals) {
            common.predictionIntervals(predictionIntervals);
            return this;
        }

        public Builder returnDiagnostics(boolean returnDiagnostics) {
            common.returnDiagnostics(returnDiagnostics);
            return this;
        }

        public Builder returnResiduals(boolean returnResiduals) {
            common.returnResiduals(returnResiduals);
            return this;
        }

        public Builder returnRobustnessWeights(boolean returnRobustnessWeights) {
            common.returnRobustnessWeights(returnRobustnessWeights);
            return this;
        }

        public Builder parallel(boolean parallel) {
            common.parallel(parallel);
            return this;
        }

        /**
         * Number of points processed per chunk (default {@code 1000}).
         */
        public Builder chunkSize(int chunkSize) {
            this.chunkSize = chunkSize;
            return this;
        }

        /**
         * Number of points overlapped between consecutive chunks (default
         * {@code 0}).
         */
        public Builder overlap(int overlap) {
            this.overlap = overlap;
            return this;
        }

        /**
         * One of
         * {@code "weighted_average"}, {@code "last_wins"}, {@code "first_wins"}
         * (default {@code "weighted_average"}).
         */
        public Builder mergeStrategy(String mergeStrategy) {
            this.mergeStrategy = mergeStrategy;
            return this;
        }

        public StreamingOptions build() {
            return new StreamingOptions(this);
        }
    }
}
