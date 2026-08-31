// Package fastlowess provides Go bindings for fastLowess, a high-performance
// implementation of LOWESS (Locally Weighted Scatterplot Smoothing) written
// in Rust, exposed to Go via cgo.
//
// Three model types are available, matching the same three offered by every
// other fastLowess binding:
//
//   - Lowess: batch smoothing. Processes an entire dataset at once and
//     supports every feature (confidence/prediction intervals,
//     cross-validation, GPU backend). Best when the dataset fits in memory.
//   - StreamingLowess: chunked processing, for datasets that don't fit in
//     memory or arrive in chunks.
//   - OnlineLowess: point-by-point processing, for real-time data.
//
// # Quickstart
//
//	opts := fastlowess.DefaultOptions()
//	opts.Fraction = 0.2
//	model, err := fastlowess.NewLowess(opts)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer model.Close()
//
//	result, err := model.Fit(x, y)
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Println(result.Y)
//
// # Building
//
// This package uses cgo to link against the native fastlowess_go library
// (built from the sibling Rust crate in this same directory). Within this
// monorepo, `make go` builds the Rust library before running `go build`/`go
// test`. Outside the monorepo, point CGO_CFLAGS/CGO_LDFLAGS at a prebuilt
// copy of the library and header (see README.md).
//
// # Resource management
//
// Lowess, StreamingLowess, and OnlineLowess all hold native (non-Go-GC-
// visible) memory and must be released with Close when no longer needed.
// A finalizer is registered as a safety net, but relying on the garbage
// collector delays releasing native memory - call Close explicitly (e.g.
// via defer) instead.
package fastlowess
