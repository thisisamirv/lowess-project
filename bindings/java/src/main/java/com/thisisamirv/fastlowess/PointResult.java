package com.thisisamirv.fastlowess;

import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * A single smoothed output produced by
 * {@link OnlineLowess#addPoint(double, double)}.
 */
public record PointResult(
        double y,
        OptionalDouble standardError,
        OptionalDouble residual,
        OptionalDouble robustnessWeight,
        OptionalInt iterationsUsed) {

    static PointResult fromNative(NativeOnlineOutput o) {
        return new PointResult(
                o.y,
                optionalDouble(o.standardError),
                optionalDouble(o.residual),
                optionalDouble(o.robustnessWeight),
                o.iterationsUsed < 0 ? OptionalInt.empty() : OptionalInt.of(o.iterationsUsed));
    }

    private static OptionalDouble optionalDouble(double value) {
        return Double.isNaN(value) ? OptionalDouble.empty() : OptionalDouble.of(value);
    }
}
