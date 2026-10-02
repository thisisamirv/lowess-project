package fastlowess;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

class OnlineLowessTest {

    @Test
    void rejectsNegativeIterations() {
        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> new OnlineLowess(OnlineOptions.builder().iterations(-1).build()));
        assertTrue(ex.getMessage() != null && !ex.getMessage().isEmpty());
    }

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
                        .intervals(IntervalsOptions.builder().confidence(0.95).prediction(0.95).build())
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

    @Test
    void bootstrapIntervalsAreSeedReproducible() {
        double[][] runs = new double[2][];
        for (int run = 0; run < 2; run++) {
            try (OnlineLowess model = new OnlineLowess(
                    OnlineOptions.builder()
                            .fraction(0.5)
                            .windowCapacity(20)
                            .minPoints(5)
                            .updateMode("full")
                            .intervals(IntervalsOptions.builder()
                                    .confidence(0.95).prediction(0.95).bootstrap(20).build())
                            .seed(0)
                            .build())) {
                Optional<PointResult> last = Optional.empty();
                for (int i = 0; i < 12; i++) {
                    Optional<PointResult> point = model.addPoint(i * 0.1, Math.sin(i * 0.1) + 0.1 * Math.cos(7 * i));
                    if (point.isPresent()) {
                        last = point;
                    }
                }
                PointResult result = last.orElseThrow();
                assertTrue(result.predictionLower().isPresent());
                runs[run] = new double[]{result.confidenceLower().orElseThrow(), result.predictionUpper().orElseThrow()};
            }
        }
        assertArrayEquals(runs[0], runs[1]);
    }

    @Test
    void bootstrapRequiresFullMode() {
        RuntimeException ex = assertThrows(RuntimeException.class, () -> new OnlineLowess(
                OnlineOptions.builder()
                        .intervals(IntervalsOptions.builder().confidence(0.95).bootstrap(10).build())
                        .build()));
        assertTrue(ex.getMessage() != null && !ex.getMessage().isEmpty());
    }
}
