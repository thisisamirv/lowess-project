package fastlowess;

import java.util.Optional;

/**
 * The result of {@link PredictModel#predict}.
 *
 * @param y the predicted value for each query point
 * @param standardErrors per-point standard errors, if requested
 * @param confidenceLower lower confidence bounds, if requested
 * @param confidenceUpper upper confidence bounds, if requested
 * @param predictionLower lower prediction bounds, if requested
 * @param predictionUpper upper prediction bounds, if requested
 * @param derivative the local fit's derivative (slope) at each query point, if
 * requested
 */
public record PredictResult(
        double[] y,
        Optional<double[]> standardErrors,
        Optional<double[]> confidenceLower,
        Optional<double[]> confidenceUpper,
        Optional<double[]> predictionLower,
        Optional<double[]> predictionUpper,
        Optional<double[]> derivative) {

    static PredictResult fromNative(NativePredictResult r) {
        return new PredictResult(
                r.y,
                Optional.ofNullable(r.standardErrors),
                Optional.ofNullable(r.confidenceLower),
                Optional.ofNullable(r.confidenceUpper),
                Optional.ofNullable(r.predictionLower),
                Optional.ofNullable(r.predictionUpper),
                Optional.ofNullable(r.derivative));
    }
}
