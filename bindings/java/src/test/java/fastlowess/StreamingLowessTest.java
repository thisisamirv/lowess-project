package fastlowess;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

class StreamingLowessTest {

    @Test
    void processesChunksAndFinalizes() {
        try (StreamingLowess model = new StreamingLowess(StreamingOptions.builder().chunkSize(10).overlap(5).build())) {
            double[] x1 = new double[10];
            double[] y1 = new double[10];
            for (int i = 0; i < 10; i++) {
                x1[i] = i;
                y1[i] = i * 2.0;
            }
            Result chunk1 = model.processChunk(x1, y1);
            assertEquals(5, chunk1.x().length);

            double[] x2 = new double[10];
            double[] y2 = new double[10];
            for (int i = 0; i < 10; i++) {
                x2[i] = i + 10;
                y2[i] = (i + 10) * 2.0;
            }
            Result chunk2 = model.processChunk(x2, y2);
            assertEquals(10, chunk2.x().length);

            Result finalResult = model.finish();
            assertTrue(finalResult.x().length > 0);
        }
    }

    @Test
    void missingDropRemovesNonFiniteRows() {
        double[] x = new double[50];
        double[] y = new double[50];
        for (int i = 0; i < 50; i++) {
            x[i] = i;
            y[i] = Math.sin(i / 10.0);
        }
        y[5] = Double.NaN;

        try (StreamingLowess model = new StreamingLowess(
                StreamingOptions.builder().fraction(0.1).chunkSize(50).missing("drop").build())) {
            Result chunkResult = model.processChunk(x, y);
            Result finalResult = model.finish();
            assertEquals(x.length - 1, chunkResult.x().length + finalResult.x().length);
        }
    }
}
