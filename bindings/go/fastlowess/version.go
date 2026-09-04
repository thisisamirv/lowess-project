package fastlowess

// version is the released version of this Go binding (bindings/go), tracked
// independently of the underlying fastLowess Rust core's crate version.
const version = "3.2.1"

// Version returns the version of this Go binding package.
func Version() string {
	return version
}
