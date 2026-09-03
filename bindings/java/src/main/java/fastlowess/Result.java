package fastlowess;

import java.util.Optional;
import java.util.OptionalInt;

/**
 * The result of fitting or updating a LOWESS model.
 *
 * @param x the sorted x values
 * @param y the smoothed y values
 * @param standardErrors per-point standard errors, if computed
 * @param confidenceLower lower confidence bounds, if computed
 * @param confidenceUpper upper confidence bounds, if computed
 * @param predictionLower lower prediction bounds, if computed
 * @param predictionUpper upper prediction bounds, if computed
 * @param residuals residuals, if computed
 * @param robustnessWeights robustness weights, if computed
 * @param cvScores cross-validation scores per tested fraction, if CV was run
 * @param fractionUsed the fraction used (as set, or selected by
 * cross-validation)
 * @param iterationsUsed the number of robustness iterations actually performed,
 * if applicable
 * @param diagnostics fit diagnostics, if requested
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
