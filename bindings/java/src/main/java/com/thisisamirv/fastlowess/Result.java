package com.thisisamirv.fastlowess;

import java.util.Optional;
import java.util.OptionalInt;

/**
 * The result of fitting or updating a LOWESS model.
 */
public record Result(
        double[] x,
        double[] y,
        Optional<double[]> standardErrors,
        Optional<double[]> confidenceLower,
        Optional<double[]> confidenceUpper,
        Optional<double[]> predictionLower,
        Optional<double[]> predictionUpper,
        Optional<double[]> residuals,
        Optional<double[]> robustnessWeights,
        Optional<double[]> cvScores,
        double fractionUsed,
        OptionalInt iterationsUsed,
        Optional<Diagnostics> diagnostics) {

    static Result fromNative(NativeResult r) {
        return new Result(
                r.x,
                r.y,
                Optional.ofNullable(r.standardErrors),
                Optional.ofNullable(r.confidenceLower),
                Optional.ofNullable(r.confidenceUpper),
                Optional.ofNullable(r.predictionLower),
                Optional.ofNullable(r.predictionUpper),
                Optional.ofNullable(r.residuals),
                Optional.ofNullable(r.robustnessWeights),
                Optional.ofNullable(r.cvScores),
                r.fractionUsed,
                r.iterationsUsed < 0 ? OptionalInt.empty() : OptionalInt.of(r.iterationsUsed),
                r.hasDiagnostics ? Optional.of(Diagnostics.fromNative(r)) : Optional.empty());
    }
}
