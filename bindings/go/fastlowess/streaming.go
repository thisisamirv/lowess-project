package fastlowess

/*
#include "fastlowess_go.h"
*/
import "C"

import (
	"errors"
	"runtime"
)

// StreamingOptions configures a StreamingLowess model. Confidence/prediction
// intervals, standard errors, cross-validation, and the GPU Backend are
// Batch-only and have no effect here.
type StreamingOptions struct {
	// Fraction is the smoothing fraction, in (0, 1]. Default: 0.67.
	Fraction float64
	// Iterations is the number of robustness iterations, in [0, 1000]. Default: 3.
	Iterations int
	// Delta is the interpolation distance threshold. Nil sets it automatically
	// to 0.0 for Streaming (interpolation disabled).
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

	// ReturnDiagnostics requests fit-quality metrics (RMSE, MAE, R-squared, AIC, etc.).
	ReturnDiagnostics bool
	// ReturnResiduals requests residuals in the result.
	ReturnResiduals bool
	// ReturnRobustnessWeights requests per-point robustness weights in the result.
	ReturnRobustnessWeights bool
	// Parallel enables parallel processing. Default: true.
	Parallel bool

	// ChunkSize is the number of points processed per chunk. Default: 5000.
	ChunkSize int
	// Overlap is the number of points shared between consecutive chunks.
	// Negative means "use the library default".
	Overlap int
	// MergeStrategy controls how overlapping chunk results are combined,
	// e.g. "weighted_average" (default).
	MergeStrategy string

	// Missing is the policy for non-finite (NaN/Inf) values in each chunk:
	// "error" (default) returns an error, "drop" silently removes affected
	// rows before merging the chunk with the overlap buffer.
	Missing string
}

// DefaultStreamingOptions returns recommended defaults for streaming use.
func DefaultStreamingOptions() StreamingOptions {
	return StreamingOptions{
		Fraction:           0.67,
		Iterations:         3,
		WeightFunction:     "tricube",
		RobustnessMethod:   "bisquare",
		ScalingMethod:      "mad",
		BoundaryPolicy:     "extend",
		ZeroWeightFallback: "use_local_mean",
		Parallel:           true,
		ChunkSize:          5000,
		Overlap:            -1,
		MergeStrategy:      "weighted_average",
		Missing:            "error",
	}
}

// StreamingLowess processes data in chunks, useful for datasets that don't
// fit in memory or arrive incrementally.
//
// StreamingLowess is not safe for concurrent use.
type StreamingLowess struct {
	ptr *C.fastlowess_GoStreamingLowess
}

// NewStreamingLowess creates a new streaming model with the given options.
func NewStreamingLowess(opts StreamingOptions) (*StreamingLowess, error) {
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
	ms := cStringOrNil(opts.MergeStrategy)
	defer freeCString(ms)
	missing := cStringOrNil(opts.Missing)
	defer freeCString(missing)

	delta, deltaSet := optPtr(opts.Delta)
	autoConverge, autoConvergeSet := optPtr(opts.AutoConverge)

	var ptr *C.fastlowess_GoStreamingLowess
	var errMsg string
	withLockedThread(func() {
		ptr = C.go_streaming_new(
			C.double(opts.Fraction),
			C.int(opts.Iterations),
			optFloat(delta, deltaSet),
			wf, rm, sm, bp,
			boolToCInt(opts.ReturnDiagnostics),
			boolToCInt(opts.ReturnResiduals),
			boolToCInt(opts.ReturnRobustnessWeights),
			zwf,
			optFloat(autoConverge, autoConvergeSet),
			boolToCInt(opts.Parallel),
			C.int(opts.ChunkSize),
			C.int(opts.Overlap),
			ms,
			missing,
		)
		if ptr == nil {
			errMsg = lastError()
		}
	})
	if ptr == nil {
		return nil, errors.New(errMsg)
	}

	s := &StreamingLowess{ptr: ptr}
	runtime.SetFinalizer(s, finalizeStreaming)
	return s, nil
}

func finalizeStreaming(s *StreamingLowess) {
	_ = s.Close()
}

// ProcessChunk fits and returns the result for one chunk of data.
func (s *StreamingLowess) ProcessChunk(x, y []float64) (Result, error) {
	if s == nil || s.ptr == nil {
		return Result{}, errors.New("fastlowess: ProcessChunk called on a closed StreamingLowess model")
	}
	if len(x) == 0 || len(x) != len(y) {
		return Result{}, errors.New("fastlowess: x and y must be non-empty and the same length")
	}
	xPtr, xLen := cDoubles(x)
	yPtr, _ := cDoubles(y)
	cres := C.go_streaming_process(s.ptr, xPtr, yPtr, xLen)
	return resultFromC(cres)
}

// Finalize flushes any buffered data and returns the final merged result.
func (s *StreamingLowess) Finalize() (Result, error) {
	if s == nil || s.ptr == nil {
		return Result{}, errors.New("fastlowess: Finalize called on a closed StreamingLowess model")
	}
	cres := C.go_streaming_finalize(s.ptr)
	return resultFromC(cres)
}

// Close releases the native resources held by this model. Safe to call
// multiple times.
func (s *StreamingLowess) Close() error {
	if s != nil && s.ptr != nil {
		C.go_streaming_free(s.ptr)
		s.ptr = nil
		runtime.SetFinalizer(s, nil)
	}
	return nil
}
