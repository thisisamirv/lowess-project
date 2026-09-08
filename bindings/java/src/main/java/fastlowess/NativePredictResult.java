package fastlowess;

/**
 * Raw predict output constructed directly by the native layer. See
 * {@link PredictResult} for the public API.
 */
final class NativePredictResult {

    final double[] y;
    final double[] standardErrors;
    final double[] confidenceLower;
    final double[] confidenceUpper;
    final double[] predictionLower;
    final double[] predictionUpper;
    final double[] derivative;

    // Constructed exclusively by the native layer (JNI bypasses normal access checks).
    @SuppressWarnings("unused") // called by JNI
    NativePredictResult(
            double[] y,
            double[] standardErrors,
            double[] confidenceLower,
            double[] confidenceUpper,
            double[] predictionLower,
            double[] predictionUpper,
            double[] derivative) {
        this.y = y;
        this.standardErrors = standardErrors;
        this.confidenceLower = confidenceLower;
        this.confidenceUpper = confidenceUpper;
        this.predictionLower = predictionLower;
        this.predictionUpper = predictionUpper;
        this.derivative = derivative;
    }
}
