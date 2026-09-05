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

    @Test
    void returnSortedDefaultsToOriginalOrder() {
        double[] x = {3.0, 1.0, 5.0, 2.0, 4.0};
        double[] y = {6.0, 2.0, 10.0, 4.0, 8.0};

        try (Lowess model = new Lowess(Options.builder().fraction(0.7).build())) {
            Result result = model.fit(x, y);
            assertTrue(java.util.Arrays.equals(result.x(), x));
        }
    }

    @Test
    void returnSortedTrueReturnsAscendingByX() {
        double[] x = {3.0, 1.0, 5.0, 2.0, 4.0};
        double[] y = {6.0, 2.0, 10.0, 4.0, 8.0};

        Options sortedOptions = Options.builder()
                .fraction(0.7)
                .returnResiduals(true)
                .returnRobustnessWeights(true)
                .returnSorted(true)
                .build();

        try (Lowess sortedModel = new Lowess(sortedOptions)) {
            Result result = sortedModel.fit(x, y);

            for (int i = 1; i < result.x().length; i++) {
                assertTrue(result.x()[i - 1] <= result.x()[i], "x should be ascending");
            }
            assertTrue(!java.util.Arrays.equals(result.x(), x),
                    "sorted x should differ from unsorted input order");

            Options unsortedOptions = Options.builder()
                    .fraction(0.7)
                    .returnResiduals(true)
                    .returnRobustnessWeights(true)
                    .build();
            try (Lowess unsortedModel = new Lowess(unsortedOptions)) {
                Result unsortedResult = unsortedModel.fit(x, y);

                double[][] sortedPairs = new double[result.x().length][2];
                for (int i = 0; i < result.x().length; i++) {
                    sortedPairs[i] = new double[]{result.x()[i], result.y()[i]};
                }
                double[][] unsortedPairs = new double[unsortedResult.x().length][2];
                for (int i = 0; i < unsortedResult.x().length; i++) {
                    unsortedPairs[i] = new double[]{unsortedResult.x()[i], unsortedResult.y()[i]};
                }
                java.util.Comparator<double[]> byX = (a, b) -> Double.compare(a[0], b[0]);
                java.util.Arrays.sort(sortedPairs, byX);
                java.util.Arrays.sort(unsortedPairs, byX);
                assertTrue(java.util.Arrays.deepEquals(sortedPairs, unsortedPairs),
                        "return_sorted should not change fitted values, only their order");
            }

            assertEquals(x.length, result.residuals().orElseThrow().length);
            assertEquals(x.length, result.robustnessWeights().orElseThrow().length);
        }
    }

    @Test
    void missingDefaultThrowsOnNan() {
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] y = {2.0, Double.NaN, 6.0, 8.0, 10.0};

        try (Lowess model = new Lowess(Options.builder().fraction(0.5).build())) {
            RuntimeException ex = org.junit.jupiter.api.Assertions.assertThrows(
                    RuntimeException.class, () -> model.fit(x, y));
            assertNotNull(ex);
        }
    }

    @Test
    void missingDropRemovesNonFiniteRows() {
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] y = {2.0, Double.NaN, 6.0, 8.0, 10.0};

        try (Lowess model = new Lowess(Options.builder().fraction(0.5).missing("drop").build())) {
            Result result = model.fit(x, y);
            assertEquals(x.length - 1, result.y().length);
        }
    }

    @Test
    void missingInvalidPolicyThrows() {
        Options opts = Options.builder().fraction(0.5).missing("invalid").build();
        RuntimeException ex = org.junit.jupiter.api.Assertions.assertThrows(
                RuntimeException.class, () -> new Lowess(opts));
        assertNotNull(ex);
    }
}
