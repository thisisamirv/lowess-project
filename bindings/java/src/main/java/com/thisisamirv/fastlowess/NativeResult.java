package com.thisisamirv.fastlowess;

/**
 * Raw fit output constructed directly by the native layer. See {@link Result}
 * for the public API.
 */
final class NativeResult {

    final double[] x;
    final double[] y;
    final double[] standardErrors;
    final double[] confidenceLower;
    final double[] confidenceUpper;
    final double[] predictionLower;
    final double[] predictionUpper;
    final double[] residuals;
    final double[] robustnessWeights;
    final double[] cvScores;
    final double fractionUsed;
    final int iterationsUsed;
    final double rmse;
    final double mae;
    final double rSquared;
    final double aic;
    final double aicc;
    final double effectiveDf;
    final double residualSd;
    final boolean hasDiagnostics;

    // Constructed exclusively by the native layer (JNI bypasses normal access checks).
    NativeResult(
            double[] x,
            double[] y,
            double[] standardErrors,
            double[] confidenceLower,
            double[] confidenceUpper,
            double[] predictionLower,
            double[] predictionUpper,
            double[] residuals,
            double[] robustnessWeights,
            double[] cvScores,
            double fractionUsed,
            int iterationsUsed,
            double rmse,
            double mae,
            double rSquared,
            double aic,
            double aicc,
            double effectiveDf,
            double residualSd,
            boolean hasDiagnostics) {
        this.x = x;
        this.y = y;
        this.standardErrors = standardErrors;
        this.confidenceLower = confidenceLower;
        this.confidenceUpper = confidenceUpper;
        this.predictionLower = predictionLower;
        this.predictionUpper = predictionUpper;
        this.residuals = residuals;
        this.robustnessWeights = robustnessWeights;
        this.cvScores = cvScores;
        this.fractionUsed = fractionUsed;
        this.iterationsUsed = iterationsUsed;
        this.rmse = rmse;
        this.mae = mae;
        this.rSquared = rSquared;
        this.aic = aic;
        this.aicc = aicc;
        this.effectiveDf = effectiveDf;
        this.residualSd = residualSd;
        this.hasDiagnostics = hasDiagnostics;
    }
}
