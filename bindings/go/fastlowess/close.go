package fastlowess

import (
	"runtime"
	"sync/atomic"
	"unsafe"
)

// closeNative swaps ptr to nil and frees the previous value exactly once,
// so an explicit Close and a GC finalizer cannot double-free.
func closeNative(ptr *unsafe.Pointer, free func(unsafe.Pointer), clearFinalizer func()) {
	if ptr == nil {
		return
	}
	p := atomic.SwapPointer(ptr, nil)
	if p != nil {
		free(p)
		if clearFinalizer != nil {
			clearFinalizer()
		}
	}
}

func loadNative(ptr *unsafe.Pointer) unsafe.Pointer {
	if ptr == nil {
		return nil
	}
	return atomic.LoadPointer(ptr)
}

func storeNative(ptr *unsafe.Pointer, value unsafe.Pointer) {
	atomic.StorePointer(ptr, value)
}

func keepAliveSlices(slices ...any) {
	for _, s := range slices {
		runtime.KeepAlive(s)
	}
}
