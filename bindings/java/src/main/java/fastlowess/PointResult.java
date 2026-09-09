package fastlowess;

import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * A single smoothed output produced by
 * {@link OnlineLowess#addPoint(double, double)}.
 *
 * @param y the smoothed value
 * @param standardError the standard error of the smoothed value, if computed
 * @param residual the residual (observed minus smoothed), if computed
 * @param robustnessWeight the final robustness weight applied to this point, if
 * computed
 * @param iterationsUsed the number of robustness iterations performed, if
 * applicable
 * @param derivative the local fit's derivative (slope) at this point, if
 * computed
 */
public record PointResult(
        double y,
        OptionalDouble standardError,
        OptionalDouble residual,
        OptionalDouble robustnessWeight,
        OptionalInt iterationsUsed,
        OptionalDouble derivative) {

    static PointResult fromNative(NativeOnlineOutput o) {
        return new PointResult(
                o.y,
                optionalDouble(o.standardError),
                optionalDouble(o.residual),
                optionalDouble(o.robustnessWeight),
                o.iterationsUsed < 0 ? OptionalInt.empty() : OptionalInt.of(o.iterationsUsed),
                optionalDouble(o.derivative));
    }

    private static OptionalDouble optionalDouble(double value) {
        return Double.isNaN(value) ? OptionalDouble.empty() : OptionalDouble.of(value);
    }
}
