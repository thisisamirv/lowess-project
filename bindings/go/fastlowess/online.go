package fastlowess

/*
#include "fastlowess_go.h"
*/
import "C"

import (
	"errors"
	"runtime"
)

// OnlineOptions configures an OnlineLowess model. Online LOWESS processes
// one point at a time, so it has no Parallel or Backend option, and
// ReturnDiagnostics/ReturnResiduals are always computed for free.
type OnlineOptions struct {
	// Fraction is the smoothing fraction, in (0, 1]. Default: 0.67.
	Fraction float64
	// Iterations is the number of robustness iterations, in [0, 1000]. Default: 3.
	Iterations int
	// Delta is the interpolation distance threshold. Nil sets it automatically
	// to 0.0 for Online (interpolation disabled).
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

	// ReturnRobustnessWeights requests per-point robustness weights in the result.
	ReturnRobustnessWeights bool

	// WindowCapacity is the maximum number of recent points retained.
	// Default: 1000.
	WindowCapacity int
	// MinPoints is the minimum number of points required before the model
	// starts producing output. Default: 2.
	MinPoints int
	// UpdateMode controls how the window is updated as new points arrive,
	// e.g. "incremental" (default).
	UpdateMode string

	// Missing is the policy for non-finite (NaN/Inf) x/y values passed to
	// AddPoint: "error" (default) returns an error, "drop" silently ignores
	// the point.
	Missing string
}

// DefaultOnlineOptions returns recommended defaults for online use.
func DefaultOnlineOptions() OnlineOptions {
	return OnlineOptions{
		Fraction:           0.67,
		Iterations:         3,
		WeightFunction:     "tricube",
		RobustnessMethod:   "bisquare",
		ScalingMethod:      "mad",
		BoundaryPolicy:     "extend",
		ZeroWeightFallback: "use_local_mean",
		WindowCapacity:     1000,
		MinPoints:          2,
		UpdateMode:         "incremental",
		Missing:            "error",
	}
}

// OnlineLowess processes one (x, y) point at a time, useful for real-time
// streaming data where results are needed immediately as points arrive.
//
// OnlineLowess is not safe for concurrent use.
type OnlineLowess struct {
	ptr *C.fastlowess_GoOnlineLowess
}

// NewOnlineLowess creates a new online model with the given options.
func NewOnlineLowess(opts OnlineOptions) (*OnlineLowess, error) {
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
	um := cStringOrNil(opts.UpdateMode)
	defer freeCString(um)
	missing := cStringOrNil(opts.Missing)
	defer freeCString(missing)

	delta, deltaSet := optPtr(opts.Delta)
	autoConverge, autoConvergeSet := optPtr(opts.AutoConverge)

	var ptr *C.fastlowess_GoOnlineLowess
	var errMsg string
	withLockedThread(func() {
		ptr = C.go_online_new(
			C.double(opts.Fraction),
			C.int(opts.Iterations),
			optFloat(delta, deltaSet),
			wf, rm, sm, bp,
			boolToCInt(opts.ReturnRobustnessWeights),
			zwf,
			optFloat(autoConverge, autoConvergeSet),
			C.int(opts.WindowCapacity),
			C.int(opts.MinPoints),
			um,
			missing,
		)
		if ptr == nil {
			errMsg = lastError()
		}
	})
	if ptr == nil {
		return nil, errors.New(errMsg)
	}

	o := &OnlineLowess{ptr: ptr}
	runtime.SetFinalizer(o, finalizeOnline)
	return o, nil
}

func finalizeOnline(o *OnlineLowess) {
	_ = o.Close()
}

// AddPoint adds a single (x, y) observation. ok is false while the window is
// still filling (fewer than MinPoints seen so far); once ok is true, res
// holds the smoothed value for the most recently added point.
func (o *OnlineLowess) AddPoint(x, y float64) (res PointResult, ok bool, err error) {
	if o == nil || o.ptr == nil {
		return PointResult{}, false, errors.New("fastlowess: AddPoint called on a closed OnlineLowess model")
	}

	cout := C.go_online_add_point(o.ptr, C.double(x), C.double(y))
	if cout.error != nil {
		msg := C.GoString(cout.error)
		C.go_online_free_output(&cout)
		return PointResult{}, false, errors.New(msg)
	}
	if cout.has_value == 0 {
		return PointResult{}, false, nil
	}

	res = PointResult{
		Y:                float64(cout.y),
		StandardError:    float64(cout.standard_error),
		Residual:         float64(cout.residual),
		RobustnessWeight: float64(cout.robustness_weight),
		IterationsUsed:   int(cout.iterations_used),
	}
	return res, true, nil
}

// Close releases the native resources held by this model. Safe to call
// multiple times.
func (o *OnlineLowess) Close() error {
	if o != nil && o.ptr != nil {
		C.go_online_free(o.ptr)
		o.ptr = nil
		runtime.SetFinalizer(o, nil)
	}
	return nil
}
