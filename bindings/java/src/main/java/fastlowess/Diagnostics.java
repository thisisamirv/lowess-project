package fastlowess;

import java.util.Optional;

/**
 * Goodness-of-fit diagnostics computed for a LOWESS fit.
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
