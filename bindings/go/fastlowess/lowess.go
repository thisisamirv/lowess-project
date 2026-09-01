package fastlowess

/*
#include "fastlowess_go.h"
*/
import "C"

import (
	"errors"
	"runtime"
)

// Options configures a Lowess, StreamingLowess, or OnlineLowess model.
// Use DefaultOptions and override only the fields you need.
type Options struct {
	// Fraction is the smoothing fraction, in (0, 1]. Default: 0.67.
	Fraction float64
	// Iterations is the number of robustness iterations, in [0, 1000]. Default: 3.
	Iterations int
	// Delta is the interpolation distance threshold, as a non-negative
	// fraction of the x range; points within Delta of each other on x share
	// the same local fit. Nil sets it automatically to 1/100th of the x range.
	Delta *float64

	// WeightFunction is the kernel weight function: "tricube" (default),
	// "gaussian", "uniform" (alias "boxcar"), "cosine", "epanechnikov",
	// "biweight" (alias "bisquare"), or "triangle" (alias "triangular").
	WeightFunction string
	// RobustnessMethod is the outlier downweighting method: "bisquare"
	// (default, alias "biweight"), "huber", or "talwar".
	RobustnessMethod string
	// ScalingMethod is the residual scale estimator for robustness weights:
	// "mad" (default, alias "median_absolute_deviation"), "mar" (alias
	// "median_absolute_residual"), or "mean" (alias "mean_absolute_residual").
	ScalingMethod string
	// BoundaryPolicy is the boundary handling strategy: "extend" (default,
	// alias "pad"), "reflect" (alias "mirror"), "zero", or "noboundary"
	// (alias "none").
	BoundaryPolicy string
	// ZeroWeightFallback is the fallback policy when all robustness weights
	// drop to zero: "use_local_mean" (default, aliases "local_mean", "mean"),
	// "return_original" (alias "original"), or "return_none" (alias "none").
	ZeroWeightFallback string

	// ConfidenceIntervals is the confidence level for confidence intervals,
	// in (0, 1) (e.g. 0.95). Nil disables confidence intervals.
	ConfidenceIntervals *float64
	// PredictionIntervals is the confidence level for prediction intervals,
	// in (0, 1) (e.g. 0.95). Nil disables prediction intervals.
	PredictionIntervals *float64
	// AutoConverge is the convergence tolerance for early stopping of
	// robustness iterations. Nil disables early stopping.
	AutoConverge *float64

	// ReturnDiagnostics requests fit-quality metrics (RMSE, MAE, R-squared, AIC, etc.).
	ReturnDiagnostics bool
	// ReturnResiduals requests residuals in the result.
	ReturnResiduals bool
	// ReturnRobustnessWeights requests per-point robustness weights in the result.
	ReturnRobustnessWeights bool
	// ReturnSE requests hat-matrix statistics (effective degrees of freedom,
	// leverage, standard errors). Batch model only.
	ReturnSE bool

	// CVFractions is a set of candidate fractions for cross-validation.
	// Empty disables CV. Batch model only.
	CVFractions []float64
	// CVMethod is the cross-validation method: "kfold" (default) or "loocv".
	CVMethod string
	// CVK is the number of folds for k-fold CV. Default: 5.
	CVK int
	// CVSeed is the RNG seed for reproducible k-fold splits. Nil uses a
	// random seed.
	CVSeed *uint64

	// Parallel enables parallel processing. Default: true.
	Parallel bool
	// Backend selects the execution backend: "cpu" (default) or "gpu". GPU
	// support requires the native library to be built with the `gpu`
	// Cargo feature. Batch model only.
	Backend string
}

// DefaultOptions returns the library's recommended defaults. Start from this
// and override only the fields you need.
func DefaultOptions() Options {
	return Options{
		Fraction:           0.67,
		Iterations:         3,
		WeightFunction:     "tricube",
		RobustnessMethod:   "bisquare",
		ScalingMethod:      "mad",
		BoundaryPolicy:     "extend",
		ZeroWeightFallback: "use_local_mean",
		CVMethod:           "kfold",
		CVK:                5,
		Parallel:           true,
		Backend:            "cpu",
	}
}

func optPtr(p *float64) (float64, bool) {
	if p == nil {
		return 0, false
	}
	return *p, true
}

// Lowess is a stateful batch LOWESS smoothing model. It processes an entire
// dataset at once and supports every feature (confidence/prediction
// intervals, cross-validation, GPU backend).
//
// Lowess is not safe for concurrent use; each goroutine should use its own
// instance, or callers must serialize access.
type Lowess struct {
	ptr *C.fastlowess_GoLowess
}

// NewLowess creates a new batch Lowess model with the given options.
func NewLowess(opts Options) (*Lowess, error) {
	wf := cStringOrNil(opts.WeightFunction)
	defer freeCString(wf)
	rm := cStringOrNil(opts.RobustnessMethod)
	defer freeCString(rm)
	sm := cStringOrNil(opts.ScalingMethod)
	defer freeCString(sm)
	bp := cStringOrNil(opts.BoundaryPolicy)
	defer freeCString(bp)
	zwf := cStringOrNil(opts.ZeroWeightFallback)
	defer freeCString(zwf)
	cvMethod := cStringOrNil(opts.CVMethod)
	defer freeCString(cvMethod)
	backend := cStringOrNil(opts.Backend)
	defer freeCString(backend)

	ci, ciSet := optPtr(opts.ConfidenceIntervals)
	pi, piSet := optPtr(opts.PredictionIntervals)
	delta, deltaSet := optPtr(opts.Delta)
	autoConverge, autoConvergeSet := optPtr(opts.AutoConverge)
	cvFracPtr, cvFracLen := cDoubles(opts.CVFractions)

	var ptr *C.fastlowess_GoLowess
	var errMsg string
	withLockedThread(func() {
		ptr = C.go_lowess_new(
			C.double(opts.Fraction),
			C.int(opts.Iterations),
			optFloat(delta, deltaSet),
			wf, rm, sm, bp,
			optFloat(ci, ciSet),
			optFloat(pi, piSet),
			boolToCInt(opts.ReturnDiagnostics),
			boolToCInt(opts.ReturnResiduals),
			boolToCInt(opts.ReturnRobustnessWeights),
			zwf,
			optFloat(autoConverge, autoConvergeSet),
			cvFracPtr, cvFracLen,
			cvMethod,
			C.int(opts.CVK),
			boolToCInt(opts.Parallel),
			boolToCInt(opts.ReturnSE),
			backend,
		)
		if ptr == nil {
			errMsg = lastError()
		}
	})
	if ptr == nil {
		return nil, errors.New(errMsg)
	}

	if opts.CVSeed != nil {
		C.go_lowess_set_cv_seed(ptr, C.ulong(*opts.CVSeed))
	}

	l := &Lowess{ptr: ptr}
	runtime.SetFinalizer(l, finalizeLowess)
	return l, nil
}

func finalizeLowess(l *Lowess) {
	_ = l.Close()
}

// Fit smooths y as a function of x. An optional customWeights slice (same
// length as x/y) applies per-observation case weights.
func (l *Lowess) Fit(x, y []float64, customWeights ...[]float64) (Result, error) {
	if l == nil || l.ptr == nil {
		return Result{}, errors.New("fastlowess: Fit called on a closed Lowess model")
	}
	if len(x) == 0 || len(x) != len(y) {
		return Result{}, errors.New("fastlowess: x and y must be non-empty and the same length")
	}
	var cw []float64
	if len(customWeights) > 0 {
		cw = customWeights[0]
	}

	xPtr, xLen := cDoubles(x)
	yPtr, _ := cDoubles(y)
	cwPtr, cwLen := cDoubles(cw)

	cres := C.go_lowess_fit(l.ptr, xPtr, yPtr, xLen, cwPtr, cwLen)
	return resultFromC(cres)
}

// Close releases the native resources held by this model. It is safe to
// call Close multiple times, and Close is called automatically by the
// garbage collector if not called explicitly, but relying on that delays
// releasing native memory - call Close explicitly (e.g. via defer) instead.
func (l *Lowess) Close() error {
	if l != nil && l.ptr != nil {
		C.go_lowess_free(l.ptr)
		l.ptr = nil
		runtime.SetFinalizer(l, nil)
	}
	return nil
}
