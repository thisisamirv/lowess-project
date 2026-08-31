package fastlowess_test

import (
	"math"
	"testing"

	"github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func sineData(n int) (x, y []float64) {
	x = make([]float64, n)
	y = make([]float64, n)
	for i := 0; i < n; i++ {
		xi := float64(i) / float64(n-1) * 10
		x[i] = xi
		// Deterministic "noise" (no RNG dependency) so tests are reproducible.
		noise := 0.05 * math.Sin(float64(i)*12.9898)
		y[i] = math.Sin(xi) + noise
	}
	return x, y
}

func linearData(n int, slope, intercept float64) (x, y []float64) {
	x = make([]float64, n)
	y = make([]float64, n)
	for i := 0; i < n; i++ {
		xi := float64(i)
		x[i] = xi
		y[i] = slope*xi + intercept
	}
	return x, y
}

func allFinite(v []float64) bool {
	for _, f := range v {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return false
		}
	}
	return true
}

func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func fitOrFatal(t *testing.T, opts fastlowess.Options, x, y []float64) fastlowess.Result {
	t.Helper()
	model, err := fastlowess.NewLowess(opts)
	if err != nil {
		t.Fatalf("NewLowess failed: %v", err)
	}
	defer model.Close()
	res, err := model.Fit(x, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	return res
}

func TestVersionAndGPUEnabled(t *testing.T) {
	if v := fastlowess.Version(); v == "" {
		t.Fatal("Version() returned an empty string")
	}
	// Just exercise the call; whether GPU is enabled depends on build flags.
	_ = fastlowess.GPUEnabled()
}

// ---------------------------------------------------------------------------
// Lowess (batch)
// ---------------------------------------------------------------------------

func TestLowess(t *testing.T) {
	t.Run("BasicSmooth", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.1, 5.9, 8.2, 9.8}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		res := fitOrFatal(t, opts, x, y)

		if len(res.Y) != len(x) || len(res.X) != len(x) {
			t.Fatalf("expected %d values, got Y=%d X=%d", len(x), len(res.Y), len(res.X))
		}
		if !approxEqual(res.FractionUsed, 0.5, 1e-9) {
			t.Fatalf("expected FractionUsed=0.5, got %v", res.FractionUsed)
		}
	})

	t.Run("BasicSmoothSerial", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.1, 5.9, 8.2, 9.8}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		opts.Parallel = false
		res := fitOrFatal(t, opts, x, y)

		if len(res.Y) != len(x) {
			t.Fatalf("expected %d values, got %d", len(x), len(res.Y))
		}
	})

	t.Run("WithDiagnostics", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.1, 5.9, 8.2, 9.8}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		opts.ReturnDiagnostics = true
		res := fitOrFatal(t, opts, x, y)

		if res.Diagnostics == nil {
			t.Fatal("expected Diagnostics to be populated")
		}
		diag := res.Diagnostics
		if diag.RMSE < 0 || diag.MAE < 0 || diag.ResidualSD < 0 {
			t.Fatalf("expected non-negative error metrics, got %+v", diag)
		}
		if diag.RSquared < 0 || diag.RSquared > 1 {
			t.Fatalf("expected RSquared in [0, 1], got %v", diag.RSquared)
		}
	})

	t.Run("WithResiduals", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.1, 5.9, 8.2, 9.8}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		opts.ReturnResiduals = true
		res := fitOrFatal(t, opts, x, y)

		if len(res.Residuals) != len(x) {
			t.Fatalf("expected %d residuals, got %d", len(x), len(res.Residuals))
		}
	})

	t.Run("WithRobustnessWeights", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.1, 100.0, 8.2, 9.8} // Outlier

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.7
		opts.Iterations = 3
		opts.ReturnRobustnessWeights = true
		res := fitOrFatal(t, opts, x, y)

		if len(res.RobustnessWeights) != len(x) {
			t.Fatalf("expected %d robustness weights, got %d", len(x), len(res.RobustnessWeights))
		}
		for _, w := range res.RobustnessWeights {
			if w < 0 || w > 1 {
				t.Fatalf("expected robustness weight in [0, 1], got %v", w)
			}
		}
	})

	t.Run("WithConfidenceIntervals", func(t *testing.T) {
		x, y := linearData(20, 2.0, 0.0)

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		ci := 0.95
		opts.ConfidenceIntervals = &ci
		res := fitOrFatal(t, opts, x, y)

		if len(res.ConfidenceLower) != len(x) || len(res.ConfidenceUpper) != len(x) {
			t.Fatal("expected confidence interval bounds to be populated")
		}
		for i := range x {
			if res.ConfidenceLower[i] > res.ConfidenceUpper[i] {
				t.Fatalf("expected lower <= upper at i=%d, got %v > %v", i, res.ConfidenceLower[i], res.ConfidenceUpper[i])
			}
		}
	})

	t.Run("WithPredictionIntervals", func(t *testing.T) {
		x, y := linearData(20, 2.0, 0.0)

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		pi := 0.95
		opts.PredictionIntervals = &pi
		res := fitOrFatal(t, opts, x, y)

		if len(res.PredictionLower) != len(x) || len(res.PredictionUpper) != len(x) {
			t.Fatal("expected prediction interval bounds to be populated")
		}
	})

	t.Run("DifferentWeightFunctions", func(t *testing.T) {
		x, y := sineData(20)

		for _, kernel := range []string{"tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle"} {
			t.Run(kernel, func(t *testing.T) {
				opts := fastlowess.DefaultOptions()
				opts.Fraction = 0.5
				opts.WeightFunction = kernel
				res := fitOrFatal(t, opts, x, y)
				if len(res.Y) != len(x) {
					t.Fatalf("expected %d values, got %d", len(x), len(res.Y))
				}
			})
		}
	})

	t.Run("DifferentRobustnessMethods", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.0, 100.0, 8.0, 10.0} // Outlier

		for _, method := range []string{"bisquare", "huber", "talwar"} {
			t.Run(method, func(t *testing.T) {
				opts := fastlowess.DefaultOptions()
				opts.Fraction = 0.7
				opts.Iterations = 3
				opts.RobustnessMethod = method
				res := fitOrFatal(t, opts, x, y)
				if len(res.Y) != len(x) {
					t.Fatalf("expected %d values, got %d", len(x), len(res.Y))
				}
			})
		}
	})

	t.Run("WithDelta", func(t *testing.T) {
		x, y := sineData(200)
		for i := range x {
			x[i] *= 10
			y[i] = math.Sin(x[i] / 10)
		}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.1
		delta := 0.1
		opts.Delta = &delta
		res := fitOrFatal(t, opts, x, y)
		if len(res.Y) != len(x) {
			t.Fatalf("expected %d values, got %d", len(x), len(res.Y))
		}
	})

	t.Run("Iterations", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.0, 100.0, 8.0, 10.0}

		for _, iterations := range []int{0, 1, 3, 5} {
			opts := fastlowess.DefaultOptions()
			opts.Fraction = 0.7
			opts.Iterations = iterations
			res := fitOrFatal(t, opts, x, y)
			if len(res.Y) != len(x) {
				t.Fatalf("iterations=%d: expected %d values, got %d", iterations, len(x), len(res.Y))
			}
		}
	})

	t.Run("Reuse", func(t *testing.T) {
		x1 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y1 := []float64{2.0, 4.1, 5.9, 8.2, 9.8}
		x2 := []float64{10.0, 20.0, 30.0, 40.0, 50.0}
		y2 := []float64{20.0, 40.0, 60.0, 80.0, 100.0}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		opts.ReturnDiagnostics = true

		model, err := fastlowess.NewLowess(opts)
		if err != nil {
			t.Fatalf("NewLowess failed: %v", err)
		}
		defer model.Close()

		res1, err := model.Fit(x1, y1)
		if err != nil {
			t.Fatalf("first Fit failed: %v", err)
		}
		res2, err := model.Fit(x2, y2)
		if err != nil {
			t.Fatalf("second Fit failed: %v", err)
		}

		if len(res1.Y) != len(x1) || len(res2.Y) != len(x2) {
			t.Fatal("expected fitted lengths to match inputs")
		}
		if res1.Diagnostics == nil || res2.Diagnostics == nil {
			t.Fatal("expected Diagnostics on both fits")
		}
	})
}

// ---------------------------------------------------------------------------
// StreamingLowess
// ---------------------------------------------------------------------------

func TestStreamingLowess(t *testing.T) {
	t.Run("ReturnsAllPoints", func(t *testing.T) {
		x, y := linearData(100, 2.0, 1.0)

		opts := fastlowess.DefaultStreamingOptions()
		opts.Fraction = 0.3
		opts.ChunkSize = 5000
		model, err := fastlowess.NewStreamingLowess(opts)
		if err != nil {
			t.Fatalf("NewStreamingLowess failed: %v", err)
		}
		defer model.Close()

		chunkRes, err := model.ProcessChunk(x, y)
		if err != nil {
			t.Fatalf("ProcessChunk failed: %v", err)
		}
		finalRes, err := model.Finalize()
		if err != nil {
			t.Fatalf("Finalize failed: %v", err)
		}

		total := len(chunkRes.Y) + len(finalRes.Y)
		if total != len(x) {
			t.Fatalf("expected %d total points, got %d", len(x), total)
		}
	})

	t.Run("Basic", func(t *testing.T) {
		x, y := sineData(2000)
		for i := range x {
			x[i] *= 100
			y[i] = math.Sin(x[i] / 100)
		}

		opts := fastlowess.DefaultStreamingOptions()
		opts.Fraction = 0.1
		opts.ChunkSize = 1000
		model, err := fastlowess.NewStreamingLowess(opts)
		if err != nil {
			t.Fatalf("NewStreamingLowess failed: %v", err)
		}
		defer model.Close()

		if _, err := model.ProcessChunk(x, y); err != nil {
			t.Fatalf("ProcessChunk failed: %v", err)
		}
		if _, err := model.Finalize(); err != nil {
			t.Fatalf("Finalize failed: %v", err)
		}
	})

	t.Run("LargerData", func(t *testing.T) {
		n := 5000
		x, y := sineData(n)
		for i := range x {
			x[i] *= 100
			y[i] = math.Sin(x[i]/100) + 0.1*math.Sin(float64(i)*7.1)
		}

		opts := fastlowess.DefaultStreamingOptions()
		opts.Fraction = 0.05
		opts.ChunkSize = 1500
		model, err := fastlowess.NewStreamingLowess(opts)
		if err != nil {
			t.Fatalf("NewStreamingLowess failed: %v", err)
		}
		defer model.Close()

		chunkRes, err := model.ProcessChunk(x, y)
		if err != nil {
			t.Fatalf("ProcessChunk failed: %v", err)
		}
		finalRes, err := model.Finalize()
		if err != nil {
			t.Fatalf("Finalize failed: %v", err)
		}

		total := len(chunkRes.Y) + len(finalRes.Y)
		if total != n {
			t.Fatalf("expected %d total points, got %d", n, total)
		}
	})

	t.Run("Accuracy", func(t *testing.T) {
		x, y := linearData(200, 2.0, 1.0)
		for i := range x {
			x[i] = float64(i) * 100.0 / 199.0
			y[i] = 2*x[i] + 1
		}

		opts := fastlowess.DefaultStreamingOptions()
		opts.Fraction = 0.5
		opts.ChunkSize = 1000
		model, err := fastlowess.NewStreamingLowess(opts)
		if err != nil {
			t.Fatalf("NewStreamingLowess failed: %v", err)
		}
		defer model.Close()

		chunkRes, err := model.ProcessChunk(x, y)
		if err != nil {
			t.Fatalf("ProcessChunk failed: %v", err)
		}
		finalRes, err := model.Finalize()
		if err != nil {
			t.Fatalf("Finalize failed: %v", err)
		}
		combined := append(append([]float64{}, chunkRes.Y...), finalRes.Y...)

		batchOpts := fastlowess.DefaultOptions()
		batchOpts.Fraction = 0.5
		batchRes := fitOrFatal(t, batchOpts, x, y)

		if len(combined) != len(batchRes.Y) {
			t.Fatalf("expected combined length %d, got %d", len(batchRes.Y), len(combined))
		}
		for i := range combined {
			if !approxEqual(combined[i], batchRes.Y[i], 1e-6) {
				t.Fatalf("streaming vs batch mismatch at i=%d: %v vs %v", i, combined[i], batchRes.Y[i])
			}
		}
	})

	t.Run("Residuals", func(t *testing.T) {
		x, y := sineData(200)
		for i := range x {
			x[i] *= 10
			y[i] = math.Sin(x[i] / 10)
		}

		opts := fastlowess.DefaultStreamingOptions()
		opts.Fraction = 0.1
		opts.ChunkSize = 50
		opts.ReturnResiduals = true
		model, err := fastlowess.NewStreamingLowess(opts)
		if err != nil {
			t.Fatalf("NewStreamingLowess failed: %v", err)
		}
		defer model.Close()

		chunkRes, err := model.ProcessChunk(x, y)
		if err != nil {
			t.Fatalf("ProcessChunk failed: %v", err)
		}
		finalRes, err := model.Finalize()
		if err != nil {
			t.Fatalf("Finalize failed: %v", err)
		}
		if chunkRes.Residuals == nil && finalRes.Residuals == nil {
			t.Fatal("expected residuals to be populated on chunk or final result")
		}
	})

	t.Run("ZeroWeightFallback", func(t *testing.T) {
		x, y := sineData(200)
		for i := range x {
			x[i] *= 10
			y[i] = math.Sin(x[i])
		}

		opts := fastlowess.DefaultStreamingOptions()
		opts.Fraction = 0.1
		opts.ChunkSize = 50
		opts.ZeroWeightFallback = "return_original"
		model, err := fastlowess.NewStreamingLowess(opts)
		if err != nil {
			t.Fatalf("NewStreamingLowess failed: %v", err)
		}
		defer model.Close()

		chunkRes, err := model.ProcessChunk(x, y)
		if err != nil {
			t.Fatalf("ProcessChunk failed: %v", err)
		}
		finalRes, err := model.Finalize()
		if err != nil {
			t.Fatalf("Finalize failed: %v", err)
		}
		total := len(chunkRes.Y) + len(finalRes.Y)
		if total != len(x) {
			t.Fatalf("expected %d total points, got %d", len(x), total)
		}
	})
}

// ---------------------------------------------------------------------------
// OnlineLowess
// ---------------------------------------------------------------------------

func TestOnlineLowess(t *testing.T) {
	t.Run("ZeroWeightFallback", func(t *testing.T) {
		opts := fastlowess.DefaultOnlineOptions()
		opts.Fraction = 0.5
		opts.WindowCapacity = 10
		opts.ZeroWeightFallback = "return_none"
		model, err := fastlowess.NewOnlineLowess(opts)
		if err != nil {
			t.Fatalf("NewOnlineLowess failed: %v", err)
		}
		defer model.Close()

		for i := 0; i < 20; i++ {
			if _, _, err := model.AddPoint(float64(i), float64(i)); err != nil {
				t.Fatalf("AddPoint failed at i=%d: %v", i, err)
			}
		}
	})

	t.Run("Basic", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
		y := []float64{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0}

		opts := fastlowess.DefaultOnlineOptions()
		opts.Fraction = 0.5
		opts.WindowCapacity = 10
		opts.MinPoints = 3
		model, err := fastlowess.NewOnlineLowess(opts)
		if err != nil {
			t.Fatalf("NewOnlineLowess failed: %v", err)
		}
		defer model.Close()

		count := 0
		for i := range x {
			_, ok, err := model.AddPoint(x[i], y[i])
			if err != nil {
				t.Fatalf("AddPoint failed at i=%d: %v", i, err)
			}
			if ok {
				count++
			}
		}
		if count == 0 {
			t.Fatal("expected at least one point result")
		}
	})

	t.Run("WithNoise", func(t *testing.T) {
		x, y := linearData(50, 2.0, 0.0)
		for i := range x {
			x[i] = float64(i) * 20.0 / 49.0
			y[i] = 2*x[i] + 0.1*math.Sin(float64(i)*3.7)
		}

		opts := fastlowess.DefaultOnlineOptions()
		opts.Fraction = 0.3
		opts.WindowCapacity = 20
		opts.MinPoints = 5
		model, err := fastlowess.NewOnlineLowess(opts)
		if err != nil {
			t.Fatalf("NewOnlineLowess failed: %v", err)
		}
		defer model.Close()

		count := 0
		for i := range x {
			_, ok, err := model.AddPoint(x[i], y[i])
			if err != nil {
				t.Fatalf("AddPoint failed at i=%d: %v", i, err)
			}
			if ok {
				count++
			}
		}
		if count == 0 {
			t.Fatal("expected at least one point result")
		}
	})

	t.Run("AddPointOneAtATime", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.0, 6.0, 8.0, 10.0}

		opts := fastlowess.DefaultOnlineOptions()
		opts.Fraction = 0.5
		opts.WindowCapacity = 10
		opts.MinPoints = 3
		model, err := fastlowess.NewOnlineLowess(opts)
		if err != nil {
			t.Fatalf("NewOnlineLowess failed: %v", err)
		}
		defer model.Close()

		sawValue := false
		for i := range x {
			_, ok, err := model.AddPoint(x[i], y[i])
			if err != nil {
				t.Fatalf("AddPoint failed at i=%d: %v", i, err)
			}
			sawValue = sawValue || ok
		}
		if !sawValue {
			t.Fatal("expected at least one point result")
		}
	})

	t.Run("Diagnostics", func(t *testing.T) {
		opts := fastlowess.DefaultOnlineOptions()
		opts.Fraction = 0.5
		opts.WindowCapacity = 50
		opts.MinPoints = 5
		opts.ReturnDiagnostics = true
		opts.ReturnResiduals = true
		model, err := fastlowess.NewOnlineLowess(opts)
		if err != nil {
			t.Fatalf("NewOnlineLowess failed: %v", err)
		}
		defer model.Close()

		x, y := sineData(100)
		sawValue := false
		for i := range x {
			res, ok, err := model.AddPoint(x[i], y[i])
			if err != nil {
				t.Fatalf("AddPoint failed at i=%d: %v", i, err)
			}
			if ok {
				sawValue = true
				if math.IsNaN(res.Y) {
					t.Fatalf("unexpected NaN Y at i=%d", i)
				}
			}
		}
		if !sawValue {
			t.Fatal("expected at least one point result once the window filled")
		}
	})
}

// ---------------------------------------------------------------------------
// Result / Diagnostics
// ---------------------------------------------------------------------------

func TestResult(t *testing.T) {
	t.Run("OptionalFieldsNilWhenNotRequested", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.0, 6.0, 8.0, 10.0}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		res := fitOrFatal(t, opts, x, y)

		if res.Diagnostics != nil {
			t.Fatal("expected Diagnostics to be nil")
		}
		if res.Residuals != nil {
			t.Fatal("expected Residuals to be nil")
		}
		if res.RobustnessWeights != nil {
			t.Fatal("expected RobustnessWeights to be nil")
		}
		if res.ConfidenceLower != nil || res.ConfidenceUpper != nil {
			t.Fatal("expected confidence bounds to be nil")
		}
		if res.PredictionLower != nil || res.PredictionUpper != nil {
			t.Fatal("expected prediction bounds to be nil")
		}
	})
}

func TestDiagnosticsValues(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	y := []float64{2.0, 4.0, 6.0, 8.0, 10.0} // Perfect linear

	opts := fastlowess.DefaultOptions()
	opts.Fraction = 0.5
	opts.ReturnDiagnostics = true
	res := fitOrFatal(t, opts, x, y)

	diag := res.Diagnostics
	if diag == nil {
		t.Fatal("expected Diagnostics to be populated")
	}
	if diag.RMSE >= 0.1 {
		t.Fatalf("expected low RMSE for perfect linear data, got %v", diag.RMSE)
	}
	if diag.MAE >= 0.1 {
		t.Fatalf("expected low MAE for perfect linear data, got %v", diag.MAE)
	}
	if diag.RSquared <= 0.99 {
		t.Fatalf("expected high RSquared for perfect linear data, got %v", diag.RSquared)
	}
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

func TestErrorHandling(t *testing.T) {
	t.Run("InvalidFractionHigh", func(t *testing.T) {
		opts := fastlowess.DefaultOptions()
		opts.Fraction = 1.5
		model, err := fastlowess.NewLowess(opts)
		if err == nil {
			defer model.Close()
			if _, err := model.Fit([]float64{1, 2, 3}, []float64{2, 4, 6}); err == nil {
				t.Fatal("expected an error for fraction > 1")
			}
		}
	})

	t.Run("InvalidFractionLow", func(t *testing.T) {
		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.0
		model, err := fastlowess.NewLowess(opts)
		if err == nil {
			defer model.Close()
			if _, err := model.Fit([]float64{1, 2, 3}, []float64{2, 4, 6}); err == nil {
				t.Fatal("expected an error for fraction <= 0")
			}
		}
	})

	t.Run("MismatchedLengths", func(t *testing.T) {
		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		model, err := fastlowess.NewLowess(opts)
		if err != nil {
			t.Fatalf("NewLowess failed: %v", err)
		}
		defer model.Close()
		if _, err := model.Fit([]float64{1, 2, 3}, []float64{2, 4}); err == nil {
			t.Fatal("expected an error for mismatched x/y lengths")
		}
	})

	t.Run("InvalidWeightFunction", func(t *testing.T) {
		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		opts.WeightFunction = "invalid"
		if _, err := fastlowess.NewLowess(opts); err == nil {
			t.Fatal("expected an error for an invalid weight function")
		}
	})

	t.Run("InvalidRobustnessMethod", func(t *testing.T) {
		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		opts.RobustnessMethod = "invalid"
		if _, err := fastlowess.NewLowess(opts); err == nil {
			t.Fatal("expected an error for an invalid robustness method")
		}
	})

	t.Run("InvalidCVMethod", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{2.0, 4.0, 6.0, 8.0, 10.0}

		opts := fastlowess.DefaultOptions()
		opts.CVFractions = []float64{0.5}
		opts.CVMethod = "invalid"
		model, err := fastlowess.NewLowess(opts)
		if err != nil {
			t.Fatalf("NewLowess unexpectedly failed: %v", err)
		}
		defer model.Close()
		if _, err := model.Fit(x, y); err == nil {
			t.Fatal("expected Fit to reject an invalid CV method")
		}
	})
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

func TestEdgeCases(t *testing.T) {
	t.Run("TwoPoints", func(t *testing.T) {
		x := []float64{1.0, 2.0}
		y := []float64{2.0, 4.0}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 1.0
		res := fitOrFatal(t, opts, x, y)
		if len(res.Y) != 2 {
			t.Fatalf("expected 2 values, got %d", len(res.Y))
		}
	})

	t.Run("LargeDataset", func(t *testing.T) {
		n := 1000
		x, y := sineData(n)
		for i := range x {
			x[i] *= 10
			y[i] = math.Sin(x[i]/10) + 0.1*math.Sin(float64(i)*3.3)
		}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.1
		res := fitOrFatal(t, opts, x, y)
		if len(res.Y) != n {
			t.Fatalf("expected %d values, got %d", n, len(res.Y))
		}
	})

	t.Run("UnsortedInput", func(t *testing.T) {
		x := []float64{3.0, 1.0, 5.0, 2.0, 4.0}
		y := []float64{6.0, 2.0, 10.0, 4.0, 8.0}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.7
		res := fitOrFatal(t, opts, x, y)
		if len(res.Y) != 5 {
			t.Fatalf("expected 5 values, got %d", len(res.Y))
		}
	})

	t.Run("DuplicateXValues", func(t *testing.T) {
		x := []float64{1.0, 1.0, 2.0, 2.0, 3.0}
		y := []float64{2.0, 2.1, 4.0, 3.9, 6.0}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.7
		res := fitOrFatal(t, opts, x, y)
		if len(res.Y) != 5 {
			t.Fatalf("expected 5 values, got %d", len(res.Y))
		}
	})

	t.Run("AllSameY", func(t *testing.T) {
		x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		y := []float64{5.0, 5.0, 5.0, 5.0, 5.0}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		res := fitOrFatal(t, opts, x, y)
		for i, v := range res.Y {
			if !approxEqual(v, 5.0, 1e-9) {
				t.Fatalf("expected constant fit at i=%d, got %v", i, v)
			}
		}
	})
}

// ---------------------------------------------------------------------------
// Cross-validation
// ---------------------------------------------------------------------------

func TestCrossValidation(t *testing.T) {
	inSet := func(v float64, set []float64) bool {
		for _, s := range set {
			if approxEqual(v, s, 1e-9) {
				return true
			}
		}
		return false
	}

	t.Run("Basic", func(t *testing.T) {
		x, y := sineData(50)
		for i := range x {
			y[i] = 2*x[i] + math.Sin(x[i])
		}

		opts := fastlowess.DefaultOptions()
		opts.CVFractions = []float64{0.2, 0.3, 0.5, 0.7}
		res := fitOrFatal(t, opts, x, y)

		if !inSet(res.FractionUsed, opts.CVFractions) {
			t.Fatalf("expected FractionUsed to be one of %v, got %v", opts.CVFractions, res.FractionUsed)
		}
		if len(res.CVScores) != len(opts.CVFractions) {
			t.Fatalf("expected %d CV scores, got %d", len(opts.CVFractions), len(res.CVScores))
		}
		if len(res.Y) != len(x) {
			t.Fatalf("expected %d values, got %d", len(x), len(res.Y))
		}
	})

	t.Run("KFold", func(t *testing.T) {
		x, y := sineData(30)
		for i := range x {
			y[i] = x[i] * x[i]
		}

		opts := fastlowess.DefaultOptions()
		opts.CVFractions = []float64{0.3, 0.5}
		opts.CVMethod = "kfold"
		opts.CVK = 5
		res := fitOrFatal(t, opts, x, y)

		if !inSet(res.FractionUsed, opts.CVFractions) {
			t.Fatalf("expected FractionUsed to be one of %v, got %v", opts.CVFractions, res.FractionUsed)
		}
		if res.CVScores == nil {
			t.Fatal("expected CVScores to be populated")
		}
	})

	t.Run("LOOCV", func(t *testing.T) {
		x, y := sineData(20)

		opts := fastlowess.DefaultOptions()
		opts.CVFractions = []float64{0.4, 0.6}
		opts.CVMethod = "loocv"
		res := fitOrFatal(t, opts, x, y)

		if !inSet(res.FractionUsed, opts.CVFractions) {
			t.Fatalf("expected FractionUsed to be one of %v, got %v", opts.CVFractions, res.FractionUsed)
		}
		if res.CVScores == nil {
			t.Fatal("expected CVScores to be populated")
		}
	})

	t.Run("WithOtherParams", func(t *testing.T) {
		x, y := sineData(40)
		for i := range x {
			y[i] = 2*x[i] + 0.5*math.Sin(x[i])
		}

		opts := fastlowess.DefaultOptions()
		opts.CVFractions = []float64{0.3, 0.5, 0.7}
		opts.Iterations = 2
		opts.ReturnDiagnostics = true
		opts.ReturnResiduals = true
		res := fitOrFatal(t, opts, x, y)

		if !inSet(res.FractionUsed, opts.CVFractions) {
			t.Fatalf("expected FractionUsed to be one of %v, got %v", opts.CVFractions, res.FractionUsed)
		}
		if res.Diagnostics == nil {
			t.Fatal("expected Diagnostics to be populated")
		}
		if res.Residuals == nil {
			t.Fatal("expected Residuals to be populated")
		}
	})

	t.Run("SingleFraction", func(t *testing.T) {
		x, y := sineData(25)
		copy(y, x)

		opts := fastlowess.DefaultOptions()
		opts.CVFractions = []float64{0.5}
		res := fitOrFatal(t, opts, x, y)

		if !approxEqual(res.FractionUsed, 0.5, 1e-9) {
			t.Fatalf("expected FractionUsed=0.5, got %v", res.FractionUsed)
		}
		if len(res.CVScores) != 1 {
			t.Fatalf("expected 1 CV score, got %d", len(res.CVScores))
		}
	})
}

// ---------------------------------------------------------------------------
// Custom weights
// ---------------------------------------------------------------------------

func TestCustomWeights(t *testing.T) {
	t.Run("UniformWeightsMatchNoWeights", func(t *testing.T) {
		x, y := sineData(20)
		weights := make([]float64, 20)
		for i := range weights {
			weights[i] = 1.0
		}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.4
		opts.Iterations = 2

		modelNoW, err := fastlowess.NewLowess(opts)
		if err != nil {
			t.Fatalf("NewLowess failed: %v", err)
		}
		defer modelNoW.Close()
		resNoW, err := modelNoW.Fit(x, y)
		if err != nil {
			t.Fatalf("Fit failed: %v", err)
		}

		modelW, err := fastlowess.NewLowess(opts)
		if err != nil {
			t.Fatalf("NewLowess failed: %v", err)
		}
		defer modelW.Close()
		resW, err := modelW.Fit(x, y, weights)
		if err != nil {
			t.Fatalf("Fit with weights failed: %v", err)
		}

		for i := range resNoW.Y {
			if !approxEqual(resNoW.Y[i], resW.Y[i], 1e-9) {
				t.Fatalf("expected matching values at i=%d, got %v vs %v", i, resNoW.Y[i], resW.Y[i])
			}
		}
	})

	t.Run("ZeroWeightReducesOutlierInfluence", func(t *testing.T) {
		n := 10
		x := make([]float64, n)
		y := make([]float64, n)
		for i := 0; i < n; i++ {
			x[i] = float64(i)
			y[i] = float64(i) * 2.0
		}
		y[5] = 100.0 // outlier

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		opts.Iterations = 0
		resNoW := fitOrFatal(t, opts, x, y)

		weights := make([]float64, n)
		for i := range weights {
			weights[i] = 1.0
		}
		weights[5] = 0.0

		model, err := fastlowess.NewLowess(opts)
		if err != nil {
			t.Fatalf("NewLowess failed: %v", err)
		}
		defer model.Close()
		resZeroW, err := model.Fit(x, y, weights)
		if err != nil {
			t.Fatalf("Fit with weights failed: %v", err)
		}

		trueVal := 5.0 * 2.0
		errNoW := math.Abs(resNoW.Y[5] - trueVal)
		errZeroW := math.Abs(resZeroW.Y[5] - trueVal)
		if errZeroW >= errNoW {
			t.Fatalf("expected zero-weighting outlier to reduce error (no_weights=%v, zero_weight=%v)", errNoW, errZeroW)
		}
	})

	t.Run("HighWeightPullsFit", func(t *testing.T) {
		n := 15
		x := make([]float64, n)
		y := make([]float64, n)
		for i := 0; i < n; i++ {
			x[i] = float64(i)
		}
		y[7] = 10.0 // spike

		weightsHigh := make([]float64, n)
		for i := range weightsHigh {
			weightsHigh[i] = 1.0
		}
		weightsHigh[7] = 100.0

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.6
		opts.Iterations = 0

		modelHigh, err := fastlowess.NewLowess(opts)
		if err != nil {
			t.Fatalf("NewLowess failed: %v", err)
		}
		defer modelHigh.Close()
		resHigh, err := modelHigh.Fit(x, y, weightsHigh)
		if err != nil {
			t.Fatalf("Fit with weights failed: %v", err)
		}

		resEqual := fitOrFatal(t, opts, x, y)

		if resHigh.Y[7] <= resEqual.Y[7] {
			t.Fatalf("expected high weight at spike to pull fit up (high=%v, equal=%v)", resHigh.Y[7], resEqual.Y[7])
		}
	})

	t.Run("CustomWeightsWithRobustness", func(t *testing.T) {
		n := 30
		x := make([]float64, n)
		y := make([]float64, n)
		weights := make([]float64, n)
		for i := 0; i < n; i++ {
			x[i] = float64(i)
			y[i] = x[i]*0.5 + math.Sin(x[i]*0.3)
			weights[i] = 1.0 + 0.1*x[i]
		}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.4
		opts.Iterations = 3

		model, err := fastlowess.NewLowess(opts)
		if err != nil {
			t.Fatalf("NewLowess failed: %v", err)
		}
		defer model.Close()
		res, err := model.Fit(x, y, weights)
		if err != nil {
			t.Fatalf("Fit with weights failed: %v", err)
		}

		if len(res.Y) != n {
			t.Fatalf("expected %d values, got %d", n, len(res.Y))
		}
		if !allFinite(res.Y) {
			t.Fatal("expected all fitted values to be finite")
		}
	})

	t.Run("WrongLengthWeightsRaiseError", func(t *testing.T) {
		n := 10
		x := make([]float64, n)
		y := make([]float64, n)
		for i := 0; i < n; i++ {
			x[i] = float64(i)
			y[i] = float64(i) * 2.0
		}
		weights := make([]float64, 7) // wrong length
		for i := range weights {
			weights[i] = 1.0
		}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		model, err := fastlowess.NewLowess(opts)
		if err != nil {
			t.Fatalf("NewLowess failed: %v", err)
		}
		defer model.Close()
		if _, err := model.Fit(x, y, weights); err == nil {
			t.Fatal("expected an error for mismatched custom_weights length")
		}
	})

	t.Run("NegativeWeightRaisesError", func(t *testing.T) {
		x := []float64{0.0, 1.0, 2.0, 3.0, 4.0}
		y := []float64{0.0, 1.0, 2.0, 3.0, 4.0}
		weights := []float64{1.0, -1.0, 1.0, 1.0, 1.0}

		opts := fastlowess.DefaultOptions()
		opts.Fraction = 0.5
		model, err := fastlowess.NewLowess(opts)
		if err != nil {
			t.Fatalf("NewLowess failed: %v", err)
		}
		defer model.Close()
		if _, err := model.Fit(x, y, weights); err == nil {
			t.Fatal("expected an error for a negative custom weight")
		}
	})
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

func TestLowessCloseIsIdempotent(t *testing.T) {
	model, err := fastlowess.NewLowess(fastlowess.DefaultOptions())
	if err != nil {
		t.Fatalf("NewLowess failed: %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("first Close failed: %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("second Close failed: %v", err)
	}
	if _, err := model.Fit([]float64{1, 2, 3}, []float64{1, 2, 3}); err == nil {
		t.Fatal("expected Fit on a closed model to return an error")
	}
}

func TestLowessInvalidIterations(t *testing.T) {
	opts := fastlowess.DefaultOptions()
	opts.Iterations = -1 // invalid: must be non-negative; validated eagerly at construction
	_, err := fastlowess.NewLowess(opts)
	if err == nil {
		t.Fatal("expected an error for negative iterations, got nil")
	}
}
