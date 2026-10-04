package fastlowess

/*
#cgo CFLAGS: -I${SRCDIR}/../include
#include "fastlowess_go.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
)

// CVOptions configures batch cross-validation. Nil disables CV.
type CVOptions struct {
	// Fractions is the candidate smoothing-fraction grid.
	Fractions []float64
	// Method is "kfold" (default) or "loocv".
	Method string
	// K is the number of k-fold splits. Ignored for loocv.
	K int
}

// IntervalsOptions groups confidence/prediction levels and residual bootstrap refits.
// Nil levels disable their respective bounds; zero Bootstrap uses analytic intervals.
type IntervalsOptions struct {
	Confidence *float64
	Prediction *float64
	Bootstrap  uint
}

// Options configures a Lowess, StreamingLowess, or OnlineLowess model.
// Use DefaultOptions and override only the fields you need. The zero value
// Options{} is not equivalent to DefaultOptions: fields such as Fraction,
// Iterations, and Parallel remain zero/false unless initialized by the helper.
type Options struct {
	// Fraction is the smoothing fraction, in (0, 1]. DefaultOptions value: 0.67.
	Fraction float64
	// Iterations is the number of robustness iterations, in [0, 1000]. DefaultOptions value: 3.
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

	// AutoConverge is the convergence tolerance for early stopping of
	// robustness iterations. Nil disables early stopping.
	AutoConverge *float64

	// Outputs selects optional result components: "diagnostics", "residuals",
	// "weights", "derivative", "se", and "sorted".
	Outputs []string
	// Intervals groups uncertainty levels and optional residual-bootstrap refits.
	Intervals *IntervalsOptions
	// CV groups cross-validation configuration. Nil disables CV.
	CV *CVOptions
	// Seed is shared by CV and fit-time bootstrap; nil uses their defaults.
	Seed *uint64

	// Parallel enables parallel processing. DefaultOptions value: true. The zero-value
	// Options struct leaves this false.
	Parallel bool
	// Backend selects the execution backend: "cpu" (default) or "gpu". GPU
	// support requires the native library to be built with the `gpu`
	// Cargo feature. Batch model only.
	Backend string

	// Missing is the policy for non-finite (NaN/Inf) values in input data:
	// "error" (default) returns an error, "drop" silently removes affected
	// observations before fitting.
	Missing string

	// RetainModel retains the fitted model's training data, enabling
	// Result.PredictModel for out-of-sample prediction. Batch model only.
	RetainModel bool
}

func hasOutput(outputs []string, name string) bool {
	for _, output := range outputs {
		if output == name {
			return true
		}
	}
	return false
}

const maxCIntValue = int64(1<<31 - 1)
const minCIntValue = -maxCIntValue - 1

func validateCInt(name string, value int) error {
	if int64(value) < minCIntValue || int64(value) > maxCIntValue {
		return fmt.Errorf("fastlowess: %s is outside the C int range", name)
	}
	return nil
}

func validateOutputs(outputs []string, adapter string, supported ...string) error {
	for _, output := range outputs {
		valid := false
		for _, name := range supported {
			if output == name {
				valid = true
				break
			}
		}
		if !valid {
			return fmt.Errorf("fastlowess: unknown %s output %q", adapter, output)
		}
	}
	return nil
}

func usesKfold(cv *CVOptions) bool {
	if cv == nil || len(cv.Fractions) == 0 {
		return false
	}
	switch strings.ToLower(cv.Method) {
	case "", "kfold", "k_fold", "k-fold":
		return true
	default:
		return false
	}
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
		Parallel:           true,
		Backend:            "cpu",
		Missing:            "error",
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
	if err := validateOutputs(opts.Outputs, "Lowess", "diagnostics", "residuals", "weights", "derivative", "se", "sorted"); err != nil {
		return nil, err
	}
	if err := validateCInt("Iterations", opts.Iterations); err != nil {
		return nil, err
	}
	if usesKfold(opts.CV) {
		folds := opts.CV.K
		if folds == 0 {
			folds = 5
		}
		if folds < 2 {
			return nil, errors.New("fastlowess: k-fold CV requires at least 2 folds")
		}
		if err := validateCInt("CV.K", folds); err != nil {
			return nil, err
		}
	}
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
	var cvMethodName string
	var cvK int
	var cvFractions []float64
	if opts.CV != nil {
		cvMethodName, cvK, cvFractions = opts.CV.Method, opts.CV.K, opts.CV.Fractions
	}
	if cvK == 0 {
		cvK = 5
	}
	cvMethod := cStringOrNil(cvMethodName)
	defer freeCString(cvMethod)
	backend := cStringOrNil(opts.Backend)
	defer freeCString(backend)
	missing := cStringOrNil(opts.Missing)
	defer freeCString(missing)

	var ci, pi *float64
	if opts.Intervals != nil {
		ci, pi = opts.Intervals.Confidence, opts.Intervals.Prediction
	}
	ciValue, ciSet := optPtr(ci)
	piValue, piSet := optPtr(pi)
	delta, deltaSet := optPtr(opts.Delta)
	autoConverge, autoConvergeSet := optPtr(opts.AutoConverge)
	cvFracPtr, cvFracLen := cDoubles(cvFractions)

	var ptr *C.fastlowess_GoLowess
	var errMsg string
	withLockedThread(func() {
		ptr = C.go_lowess_new(
			C.double(opts.Fraction),
			C.int(opts.Iterations),
			optFloat(delta, deltaSet),
			wf, rm, sm, bp,
			optFloat(ciValue, ciSet),
			optFloat(piValue, piSet),
			boolToCInt(hasOutput(opts.Outputs, "diagnostics")),
			boolToCInt(hasOutput(opts.Outputs, "residuals")),
			boolToCInt(hasOutput(opts.Outputs, "weights")),
			zwf,
			optFloat(autoConverge, autoConvergeSet),
			cvFracPtr, cvFracLen,
			cvMethod,
			C.int(cvK),
			boolToCInt(opts.Parallel),
			boolToCInt(hasOutput(opts.Outputs, "se")),
			boolToCInt(hasOutput(opts.Outputs, "sorted")),
			backend,
			missing,
			boolToCInt(opts.RetainModel),
			boolToCInt(hasOutput(opts.Outputs, "derivative")),
		)
		if ptr == nil {
			errMsg = lastError()
		}
	})
	if ptr == nil {
		return nil, errors.New(errMsg)
	}

	if opts.Seed != nil {
		C.go_lowess_set_seed(ptr, C.ulonglong(*opts.Seed))
	}
	if opts.Intervals != nil && opts.Intervals.Bootstrap > 0 {
		C.go_lowess_set_bootstrap(ptr, C.size_t(opts.Intervals.Bootstrap))
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
	if len(customWeights) > 1 {
		return Result{}, errors.New("fastlowess: Fit accepts at most one custom weight slice")
	}
	var cw []float64
	if len(customWeights) > 0 {
		cw = customWeights[0]
	}

	xPtr, xLen := cDoubles(x)
	yPtr, _ := cDoubles(y)
	cwPtr, cwLen := cDoubles(cw)

	cres := C.go_lowess_fit(l.ptr, xPtr, yPtr, xLen, cwPtr, cwLen)
	runtime.KeepAlive(l)
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
