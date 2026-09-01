package fastlowess

/*
#cgo CFLAGS: -I${SRCDIR}/../include
#cgo linux LDFLAGS: -L${SRCDIR}/../../../target/release-c -lfastlowess_go -lm -ldl -lpthread
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../target/release-c -lfastlowess_go
#cgo windows LDFLAGS: -L${SRCDIR}/../../../target/x86_64-pc-windows-gnu/release-c -lfastlowess_go -lws2_32 -luserenv -lbcrypt -lntdll -lpthread
#include <stdlib.h>
#include "fastlowess_go.h"
*/
import "C"

import (
	"errors"
	"math"
	"runtime"
	"unsafe"
)

// GPUEnabled reports whether this build of the native library was compiled
// with GPU backend support (the `gpu` Cargo feature).
func GPUEnabled() bool {
	return C.go_gpu_enabled() != 0
}

// lastError reads the thread-local error message set by the most recent
// failed constructor call. Callers MUST invoke this on the same OS thread as
// the call that may have set it - see withLockedThread.
func lastError() string {
	cmsg := C.go_last_error_message()
	if cmsg == nil {
		return "unknown error"
	}
	return C.GoString(cmsg)
}

// withLockedThread pins the calling goroutine to its current OS thread for
// the duration of f. This is required whenever we make a cgo call that may
// set fastLowess's thread-local last-error slot, followed by a second cgo
// call to read it back (go_last_error_message) - without this, Go's
// scheduler could migrate the goroutine to a different OS thread in
// between, and we'd read an unrelated thread's (empty) error slot.
func withLockedThread(f func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	f()
}

func cStringOrNil(s string) *C.char {
	if s == "" {
		return nil
	}
	return C.CString(s)
}

func freeCString(s *C.char) {
	if s != nil {
		C.free(unsafe.Pointer(s))
	}
}

func boolToCInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

func optFloat(v float64, set bool) C.double {
	if !set {
		return C.double(math.NaN())
	}
	return C.double(v)
}

// cDoubles returns a pointer to the first element of xs (or nil if empty)
// and its length, suitable for passing to a `const double *, unsigned long`
// FFI parameter pair. The backing array of a []float64 contains no Go
// pointers, so passing its address across cgo is safe per the cgo pointer
// passing rules.
func cDoubles(xs []float64) (*C.double, C.ulong) {
	if len(xs) == 0 {
		return nil, 0
	}
	return (*C.double)(unsafe.Pointer(&xs[0])), C.ulong(len(xs))
}

// cDoubleSliceToGo copies n float64s out of a Rust-allocated buffer. Returns
// nil if ptr is nil (meaning the field was not computed).
func cDoubleSliceToGo(ptr *C.double, n int) []float64 {
	if ptr == nil || n == 0 {
		return nil
	}
	src := unsafe.Slice((*float64)(unsafe.Pointer(ptr)), n)
	out := make([]float64, n)
	copy(out, src)
	return out
}

// Diagnostics holds goodness-of-fit metrics, populated when ReturnDiagnostics
// is enabled.
type Diagnostics struct {
	RMSE        float64
	MAE         float64
	RSquared    float64
	AIC         float64
	AICc        float64
	EffectiveDF float64
	ResidualSD  float64
}

// Result is the outcome of a batch fit, streaming chunk/finalize, or is
// embedded conceptually (as PointResult) for the online model.
type Result struct {
	// X is the sorted input x values (length N).
	X []float64
	// Y is the smoothed y values (length N).
	Y []float64

	// StandardErrors is nil unless ReturnSE was requested.
	StandardErrors []float64
	// ConfidenceLower/ConfidenceUpper are nil unless ConfidenceIntervals was set.
	ConfidenceLower []float64
	ConfidenceUpper []float64
	// PredictionLower/PredictionUpper are nil unless PredictionIntervals was set.
	PredictionLower []float64
	PredictionUpper []float64
	// Residuals is nil unless ReturnResiduals was requested.
	Residuals []float64
	// RobustnessWeights is nil unless ReturnRobustnessWeights was requested.
	RobustnessWeights []float64
	// CVScores is nil unless cross-validation was configured.
	CVScores []float64

	// FractionUsed is the smoothing fraction actually applied.
	FractionUsed float64
	// IterationsUsed is the number of robustness iterations performed, or -1
	// if not available (e.g. for streaming intermediate chunks).
	IterationsUsed int

	// Diagnostics is nil unless ReturnDiagnostics was requested.
	Diagnostics *Diagnostics
}

func resultFromC(cres C.fastlowess_GoLowessResult) (Result, error) {
	if cres.error != nil {
		msg := C.GoString(cres.error)
		C.go_lowess_free_result(&cres)
		return Result{}, errors.New(msg)
	}

	n := int(cres.n)
	cvN := int(cres.cv_scores_len)

	r := Result{
		X:                 cDoubleSliceToGo(cres.x, n),
		Y:                 cDoubleSliceToGo(cres.y, n),
		StandardErrors:    cDoubleSliceToGo(cres.standard_errors, n),
		ConfidenceLower:   cDoubleSliceToGo(cres.confidence_lower, n),
		ConfidenceUpper:   cDoubleSliceToGo(cres.confidence_upper, n),
		PredictionLower:   cDoubleSliceToGo(cres.prediction_lower, n),
		PredictionUpper:   cDoubleSliceToGo(cres.prediction_upper, n),
		Residuals:         cDoubleSliceToGo(cres.residuals, n),
		RobustnessWeights: cDoubleSliceToGo(cres.robustness_weights, n),
		CVScores:          cDoubleSliceToGo(cres.cv_scores, cvN),
		FractionUsed:      float64(cres.fraction_used),
		IterationsUsed:    int(cres.iterations_used),
	}

	if !math.IsNaN(float64(cres.rmse)) {
		r.Diagnostics = &Diagnostics{
			RMSE:        float64(cres.rmse),
			MAE:         float64(cres.mae),
			RSquared:    float64(cres.r_squared),
			AIC:         float64(cres.aic),
			AICc:        float64(cres.aicc),
			EffectiveDF: float64(cres.effective_df),
			ResidualSD:  float64(cres.residual_sd),
		}
	}

	C.go_lowess_free_result(&cres)
	return r, nil
}

// PointResult is the outcome of OnlineLowess.AddPoint once the window has
// enough points to produce a smoothed value.
type PointResult struct {
	Y                float64
	StandardError    float64 // NaN if not computed
	Residual         float64 // NaN if not computed
	RobustnessWeight float64 // NaN if not computed
	IterationsUsed   int     // -1 if not applicable
}
