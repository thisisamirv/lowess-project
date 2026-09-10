package fastlowess;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertThrows;
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

    @Test
    void returnSeAndIntervalsRequiresFullMode() {
        RuntimeException ex = assertThrows(RuntimeException.class, () -> new OnlineLowess(
                OnlineOptions.builder().fraction(0.5).windowCapacity(10).minPoints(3).returnSe(true).build()));
        assertTrue(ex.getMessage() != null && !ex.getMessage().isEmpty());
    }

    @Test
    void returnSeAndIntervals() {
        try (OnlineLowess model = new OnlineLowess(
                OnlineOptions.builder()
                        .fraction(0.5)
                        .windowCapacity(10)
                        .minPoints(3)
                        .updateMode("full")
                        .returnSe(true)
                        .confidenceIntervals(0.95)
                        .predictionIntervals(0.95)
                        .build())) {
            Optional<PointResult> last = Optional.empty();
            for (int i = 0; i < 10; i++) {
                Optional<PointResult> point = model.addPoint(i, i * 2.0);
                if (point.isPresent()) {
                    last = point;
                }
            }
            assertTrue(last.isPresent(), "expected at least one point result");
            PointResult result = last.get();
            assertTrue(result.standardError().isPresent());
            assertTrue(result.confidenceLower().isPresent());
            assertTrue(result.confidenceUpper().isPresent());
            assertTrue(result.predictionLower().isPresent());
            assertTrue(result.predictionUpper().isPresent());
        }
    }
}
