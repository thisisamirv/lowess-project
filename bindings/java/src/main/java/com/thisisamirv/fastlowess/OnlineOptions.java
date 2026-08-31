package com.thisisamirv.fastlowess;

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

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Fluent builder for {@link OnlineOptions}.
     */
    public static final class Builder {

        private final Options.Builder common = Options.builder();
        int windowCapacity = 100;
        int minPoints = 10;
        String updateMode = null;

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
         * Maximum number of points retained in the sliding window (default
         * {@code 100}).
         */
        public Builder windowCapacity(int windowCapacity) {
            this.windowCapacity = windowCapacity;
            return this;
        }

        /**
         * Minimum number of points required before a fit is produced (default
         * {@code 10}).
         */
        public Builder minPoints(int minPoints) {
            this.minPoints = minPoints;
            return this;
        }

        /**
         * One of {@code "incremental"}, {@code "full_refit"} (default
         * {@code "incremental"}).
         */
        public Builder updateMode(String updateMode) {
            this.updateMode = updateMode;
            return this;
        }

        public OnlineOptions build() {
            return new OnlineOptions(this);
        }
    }
}
