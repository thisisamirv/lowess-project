//! Tests for `binding_support`'s FFI buffer transfer helpers.
//!
//! These verify the ownership-transfer + free round trip used by every language
//! binding to hand heap-allocated `Vec<f64>` buffers across the FFI boundary.
//! The helper under test uses `Box::into_raw` (not `mem::forget`) and its exact
//! inverse `Box::from_raw` on the free side, so the round trip is leak-free by
//! construction; the C++ binding additionally runs these paths under Valgrind
//! (`--leak-check=full`) in CI (`make cpp-dev`, "3b. Valgrind memory check").
//!
//! Only compiled when the `dev` feature is enabled, which is what exposes the
//! `fastLowess::internals::binding_support` module.
#![cfg(feature = "dev")]

use fastLowess::internals::binding_support::{
    free_raw_f64_buffer, opt_vec_to_raw_ptr, vec_to_raw_ptr,
};

/// A `Vec` moved through `vec_to_raw_ptr` must surface identical data at the
/// returned pointer, and `free_raw_f64_buffer` must release it without panic or
/// corruption. Round-tripping repeatedly exercises the allocate/forget/free
/// cycle the reviewer flagged, proving it is a symmetric ownership transfer
/// rather than an accumulating leak.
#[test]
fn vec_to_raw_ptr_round_trips_data_and_frees() {
    let expected = vec![1.0, 2.0, 3.5, -4.0, 5.25];
    let len = expected.len();

    for _ in 0..100 {
        let ptr = vec_to_raw_ptr(expected.clone());

        // SAFETY: `ptr` was just produced by `vec_to_raw_ptr` with `len`
        // elements and is still live.
        let actual = unsafe { std::slice::from_raw_parts(ptr, len) };
        assert_eq!(
            actual,
            expected.as_slice(),
            "buffer contents were corrupted"
        );

        // SAFETY: `ptr`/`len` match what `vec_to_raw_ptr` produced.
        unsafe { free_raw_f64_buffer(ptr, len) };
    }
}

/// `opt_vec_to_raw_ptr` must return a valid pointer for `Some` (with the same
/// contents) and a null pointer for `None`.
#[test]
fn opt_vec_to_raw_ptr_some_and_none() {
    let ptr = opt_vec_to_raw_ptr(Some(vec![7.0, 8.0, 9.0]));
    assert!(!ptr.is_null());
    // SAFETY: `ptr` was produced by `opt_vec_to_raw_ptr(Some(..))` with 3 elements.
    let slice = unsafe { std::slice::from_raw_parts(ptr, 3) };
    assert_eq!(slice, &[7.0, 8.0, 9.0]);
    // SAFETY: `ptr`/len 3 match what `vec_to_raw_ptr` produced.
    unsafe { free_raw_f64_buffer(ptr, 3) };

    let null_ptr = opt_vec_to_raw_ptr(None::<Vec<f64>>);
    assert!(null_ptr.is_null());
    // SAFETY: freeing a null pointer is documented as a no-op.
    unsafe { free_raw_f64_buffer(null_ptr, 0) };
}

/// `free_raw_f64_buffer` must tolerate a null pointer without panicking.
#[test]
fn free_raw_f64_buffer_null_is_noop() {
    // SAFETY: `free_raw_f64_buffer` documents null as a no-op.
    unsafe {
        free_raw_f64_buffer(std::ptr::null_mut(), 0);
        free_raw_f64_buffer(std::ptr::null_mut(), 10);
    }
}
