package fastlowess;

import java.util.Optional;

/**
 * Goodness-of-fit diagnostics computed for a LOWESS fit.
 *
 * @param rmse root mean squared error
 * @param mae mean absolute error
 * @param rSquared R-squared
 * @param aic Akaike Information Criterion, if computed
 * @param aicc corrected Akaike Information Criterion, if computed
 * @param effectiveDf effective degrees of freedom, if computed
 * @param residualSd residual standard deviation
 */
public record Diagnostics(
        double rmse,
        double mae,
        double rSquared,
        Optional<Double> aic,
        Optional<Double> aicc,
        Optional<Double> effectiveDf,
        double residualSd) {

    static Diagnostics fromNative(NativeResult r) {
        return new Diagnostics(
                r.rmse,
                r.mae,
                r.rSquared,
                optionalDouble(r.aic),
                optionalDouble(r.aicc),
                optionalDouble(r.effectiveDf),
                r.residualSd);
    }

    private static Optional<Double> optionalDouble(double value) {
        return Double.isNaN(value) ? Optional.empty() : Optional.of(value);
    }
}
