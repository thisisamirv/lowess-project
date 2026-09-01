package fastlowess

/*
#include "fastlowess_go.h"
*/
import "C"

import (
	"errors"
	"runtime"
)

// StreamingOptions configures a StreamingLowess model.
type StreamingOptions struct {
	Options
	// ChunkSize is the number of points processed per chunk. Default: 5000.
	ChunkSize int
	// Overlap is the number of points shared between consecutive chunks.
	// Negative means "use the library default".
	Overlap int
	// MergeStrategy controls how overlapping chunk results are combined,
	// e.g. "weighted_average" (default).
	MergeStrategy string
}

// DefaultStreamingOptions returns recommended defaults for streaming use.
func DefaultStreamingOptions() StreamingOptions {
	return StreamingOptions{
		Options:       DefaultOptions(),
		ChunkSize:     5000,
		Overlap:       -1,
		MergeStrategy: "weighted_average",
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

	ci, ciSet := optPtr(opts.ConfidenceIntervals)
	pi, piSet := optPtr(opts.PredictionIntervals)
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
			optFloat(ci, ciSet),
			optFloat(pi, piSet),
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
