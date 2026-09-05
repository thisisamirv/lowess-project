package fastlowess;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

class OnlineLowessTest {

    @Test
    void addsPointsAndEventuallyProducesOutput() {
        try (OnlineLowess model = new OnlineLowess(OnlineOptions.builder().minPoints(5).build())) {
            boolean sawValue = false;
            for (int i = 0; i < 20; i++) {
                Optional<PointResult> point = model.addPoint(i, i * 2.0);
                if (point.isPresent()) {
                    sawValue = true;
                }
            }
            assertTrue(sawValue, "expected at least one point result once minPoints was reached");
        }
    }

    @Test
    void missingDropIgnoresNonFinitePoint() {
        try (OnlineLowess model = new OnlineLowess(
                OnlineOptions.builder().fraction(0.5).windowCapacity(10).missing("drop").build())) {
            Optional<PointResult> point = model.addPoint(1.0, Double.NaN);
            assertTrue(point.isEmpty(), "expected non-finite point to be ignored under missing=drop");
        }
    }
}
