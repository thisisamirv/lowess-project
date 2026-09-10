package fastlowess;

/**
 * Raw online-update output constructed directly by the native layer. See
 * {@link PointResult}.
 */
final class NativeOnlineOutput {

    final boolean hasValue;
    final double y;
    final double standardError;
    final double residual;
    final double robustnessWeight;
    final int iterationsUsed;
    final double derivative;
    final double confidenceLower;
    final double confidenceUpper;
    final double predictionLower;
    final double predictionUpper;

    @SuppressWarnings("unused") // called by JNI
    NativeOnlineOutput(
            boolean hasValue,
            double y,
            double standardError,
            double residual,
            double robustnessWeight,
            int iterationsUsed,
            double derivative,
            double confidenceLower,
            double confidenceUpper,
            double predictionLower,
            double predictionUpper) {
        this.hasValue = hasValue;
        this.y = y;
        this.standardError = standardError;
        this.residual = residual;
        this.robustnessWeight = robustnessWeight;
        this.iterationsUsed = iterationsUsed;
        this.derivative = derivative;
        this.confidenceLower = confidenceLower;
        this.confidenceUpper = confidenceUpper;
        this.predictionLower = predictionLower;
        this.predictionUpper = predictionUpper;
    }
}
