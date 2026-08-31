package fastlowess;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

class LowessTest {

    private static double[] linspace(int n) {
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = i;
        }
        return x;
    }

    @Test
    void fitsASimpleLine() {
        double[] x = linspace(20);
        double[] y = new double[20];
        for (int i = 0; i < 20; i++) {
            y[i] = x[i] * 2.0;
        }

        try (Lowess model = new Lowess(Options.builder().build())) {
            Result result = model.fit(x, y);
            assertEquals(20, result.x().length);
            assertEquals(20, result.y().length);
            for (int i = 5; i < 15; i++) {
                assertTrue(Math.abs(result.y()[i] - y[i]) < 1.0, "expected near-linear fit at index " + i);
            }
        }
    }

    @Test
    void returnsDiagnosticsWhenRequested() {
        double[] x = linspace(30);
        double[] y = new double[30];
        for (int i = 0; i < 30; i++) {
            y[i] = Math.sin(x[i] / 5.0);
        }

        try (Lowess model = new Lowess(Options.builder().returnDiagnostics(true).build())) {
            Result result = model.fit(x, y);
            assertTrue(result.diagnostics().isPresent());
            assertTrue(result.diagnostics().get().rmse() >= 0.0);
        }
    }

    @Test
    void throwsOnEmptyInput() {
        try (Lowess model = new Lowess(Options.builder().build())) {
            RuntimeException ex = org.junit.jupiter.api.Assertions.assertThrows(
                    RuntimeException.class, () -> model.fit(new double[0], new double[0]));
            assertNotNull(ex);
        }
    }

    @Test
    void throwsAfterClose() {
        Lowess model = new Lowess(Options.builder().build());
        model.close();
        IllegalStateException ex = org.junit.jupiter.api.Assertions.assertThrows(
                IllegalStateException.class, () -> model.fit(linspace(5), linspace(5)));
        assertNotNull(ex);
    }
}
