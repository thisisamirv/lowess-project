#![cfg(feature = "dev")]
//! Tests for window management utilities.
//!
//! These tests verify the window initialization and recentering logic used in LOWESS for:
//! - Window initialization with various sizes and positions
//! - Window recentering based on data distribution
//! - Boundary clamping and edge case handling
//!
//! ## Test Organization
//!
//! 1. **Window Initialization** - Basic centering, clamping, edge cases
//! 2. **Window Recentering** - Shifting, clamping, boundary conditions
//! 3. **Edge Cases** - Empty data, invalid indices, debug assertions

use lowess::internals::primitives::window::Window;

/// Extension trait for testing Window properties.
trait WindowTestExt {
    fn new(left: usize, right: usize) -> Option<Self>
    where
        Self: Sized;
}

impl WindowTestExt for Window {
    fn new(left: usize, right: usize) -> Option<Self> {
        if left <= right {
            Some(Self { left, right })
        } else {
            None
        }
    }
}

// ============================================================================
// Window Initialization Tests
// ============================================================================

/// Test basic window initialization with centering.
///
/// Verifies that window is centered around target index.
#[test]
fn test_initialize_window_basic() {
    let n = 10;
    let window_size = 5;
    let idx = 5;
    let win = Window::initialize(idx, window_size, n);

    assert_eq!(win.len(), window_size, "Window should have correct size");
    assert!(
        win.left <= idx && idx <= win.right,
        "Index should be within window"
    );
    assert!(win.right < n, "Window should be within bounds");
}

/// Test window initialization clamps near end.
///
/// Verifies that window is adjusted when near array end.
#[test]
fn test_initialize_window_near_end() {
    let n = 5;
    let window_size = 3;
    let idx = 4; // Near the end

    let win = Window::initialize(idx, window_size, n);

    assert_eq!(win.len(), window_size, "Window should have correct size");
    assert_eq!(win.right, n - 1, "Right should be at last index");
    assert_eq!(
        win.left,
        n - window_size,
        "Left should be adjusted to maintain size"
    );
}

/// Test window initialization with full range.
///
/// Verifies that when window_size >= n, entire range is used.
#[test]
fn test_initialize_window_full_range() {
    let n = 4;
    let window_size = 10; // Larger than n

    let win = Window::initialize(0, window_size, n);

    assert_eq!((win.left, win.right), (0, n - 1), "Should use full range");
}

/// Test window initialization with size one.
///
/// Verifies that size-1 window is positioned at target index.
#[test]
fn test_initialize_window_size_one() {
    let n = 7;
    let window_size = 1;

    for idx in 0..n {
        let win = Window::initialize(idx, window_size, n);
        assert_eq!(win.left, idx, "Left should equal index");
        assert_eq!(win.right, idx, "Right should equal index");
    }
}

/// Test window initialization with even window size.
///
/// Verifies correct centering behavior for even-sized windows.
#[test]
fn test_initialize_window_even_size() {
    let n = 10;
    let window_size = 4; // Even size

    for idx in 0..n {
        let win = Window::initialize(idx, window_size, n);
        assert_eq!(win.len(), window_size, "Window should have correct size");
        assert!(
            win.left <= idx && idx <= win.right,
            "Index should be within window"
        );
        assert!(win.right < n, "Window should be within bounds");
    }
}

/// Test window initialization at array start.
///
/// Verifies correct behavior when initializing at index 0.
#[test]
fn test_initialize_window_at_start() {
    let n = 10;
    let window_size = 5;
    let idx = 0;

    let win = Window::initialize(idx, window_size, n);

    assert_eq!(win.left, 0, "Left should be at start");
    assert!(
        win.right < window_size,
        "Right should be within window size"
    );
    assert!(win.left <= idx && idx <= win.right);
}

/// Test window initialization at array end.
///
/// Verifies correct behavior when initializing at last index.
#[test]
fn test_initialize_window_at_end() {
    let n = 10;
    let window_size = 5;
    let idx = n - 1;

    let win = Window::initialize(idx, window_size, n);

    assert_eq!(win.right, n - 1, "Right should be at end");
    assert_eq!(
        win.left,
        n - window_size,
        "Left should maintain window size"
    );
    assert!(win.left <= idx && idx <= win.right);
}

// ============================================================================
// Window Recentering Tests
// ============================================================================

/// Test window recentering shifts when appropriate.
///
/// Verifies that window shifts to closer data cluster.
#[test]
fn test_recenter_window_shifts() {
    // Crafted x so left side is far, right side is close
    let x = vec![0.0, 1.0, 100.0, 101.0, 102.0];
    let n = x.len();
    let current = 2; // At 100.0

    // Start with window covering [0, 2]
    let mut win = Window::new(0, 2).unwrap();
    win.recenter(&x, current, n);

    // Should shift to include close cluster [100, 101, 102]
    assert_eq!(
        (win.left, win.right),
        (2, 4),
        "Window should shift to closer points"
    );
}

/// Test window recentering clamps invalid bounds.
///
/// Verifies that out-of-range bounds are clamped safely.
#[test]
fn test_recenter_window_clamps_invalid() {
    let x = vec![0.0, 1.0, 2.0];
    let n = x.len();
    let current = 1;

    // Provide out-of-range left/right
    let mut win = Window {
        left: 100,
        right: 100,
    };
    win.recenter(&x, current, n);

    // Should clamp to valid range
    assert!(win.left < n, "Left should be clamped");
    assert!(win.right < n, "Right should be clamped");
}

/// Test window recentering with no shift needed.
///
/// Verifies that window doesn't shift when already optimal.
#[test]
fn test_recenter_window_no_shift() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let n = x.len();
    let current = 2; // Middle point

    // Window already centered
    let mut win = Window::new(1, 3).unwrap();
    win.recenter(&x, current, n);

    // Window should remain similar (may adjust slightly)
    assert!(
        win.left <= current && current <= win.right,
        "Current should remain in window"
    );
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/// Test Window::new with valid bounds.
///
/// Verifies that valid bounds create window successfully.
#[test]
fn test_window_new_valid() {
    let win = Window::new(0, 5);
    assert!(win.is_some(), "Valid bounds should succeed");

    let w = win.unwrap();
    assert_eq!(w.left, 0);
    assert_eq!(w.right, 5);
}

/// Test Window::new with invalid bounds.
///
/// Verifies that left > right produces None.
#[test]
fn test_window_new_invalid() {
    let win = Window::new(5, 0);
    assert!(
        win.is_none(),
        "Invalid bounds (left > right) should return None"
    );
}

/// Test window length calculation.
///
/// Verifies that len() returns correct window size.
#[test]
fn test_window_len() {
    let win = Window::new(2, 7).unwrap();
    assert_eq!(win.len(), 6, "Length should be right - left + 1");

    let win_single = Window::new(3, 3).unwrap();
    assert_eq!(win_single.len(), 1, "Single-point window should have len 1");
}

/// Test window initialization with various sizes.
///
/// Verifies correct behavior across different window sizes.
#[test]
fn test_initialize_various_sizes() {
    let n = 20;

    for window_size in 1..=n {
        for idx in 0..n {
            let win = Window::initialize(idx, window_size, n);

            // Verify window properties
            assert!(win.left <= win.right, "Left should be <= right");
            assert!(win.right < n, "Right should be < n");
            assert!(
                win.left <= idx && idx <= win.right,
                "Index should be in window"
            );

            // Verify size (may be smaller than requested if near boundaries)
            assert!(
                win.len() <= window_size,
                "Window should not exceed requested size"
            );
        }
    }
}
// ============================================================================
// Additional Edge Case Tests
// ============================================================================

/// Test Window::calculate_span with edge case fractions.
#[test]
fn test_calculate_span_edge_fractions() {
    let n = 100;

    // Very small fraction
    let span = Window::calculate_span(n, 0.01);
    assert_eq!(span, 2, "Minimum span should be 2");

    // Fraction of 0
    let span = Window::calculate_span(n, 0.0);
    assert_eq!(span, 2, "Zero fraction should give minimum span");

    // Fraction >= 1
    let span = Window::calculate_span(n, 1.0);
    assert_eq!(span, n, "Fraction of 1.0 should give full span");

    let span = Window::calculate_span(n, 1.5);
    assert_eq!(span, n, "Fraction > 1.0 should be clamped to n");
}

/// Test Window::calculate_span with small n.
#[test]
fn test_calculate_span_small_n() {
    // n = 1: max(2, min(1, floor(0.5 * 1))) = max(2, 0) = 2
    let span = Window::calculate_span(1, 0.5);
    assert_eq!(span, 2, "Span minimum is 2, even when n=1");

    // n = 2
    let span = Window::calculate_span(2, 0.5);
    assert_eq!(span, 2, "Span should be at least 2 when n >= 2");

    // n = 3
    let span = Window::calculate_span(3, 0.5);
    assert_eq!(span, 2, "Span should be max(2, floor(0.5 * 3)) = 2");
}

/// Test Window::max_distance with various distributions.
#[test]
fn test_max_distance_distributions() {
    // Symmetric distribution
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let win = Window::new(1, 3).unwrap();
    let dist = win.max_distance(&x, 2.0); // Center point
    assert_eq!(dist, 1.0, "Max distance from center should be 1.0");

    // Asymmetric - closer to left
    let win = Window::new(0, 2).unwrap();
    let dist = win.max_distance(&x, 1.0);
    assert_eq!(dist, 1.0, "Max distance should be max(1-0, 2-1) = 1.0");

    // Asymmetric - closer to right
    let win = Window::new(2, 4).unwrap();
    let dist = win.max_distance(&x, 3.0);
    assert_eq!(dist, 1.0, "Max distance should be max(3-2, 4-3) = 1.0");
}

/// Test Window::max_distance at boundaries.
#[test]
fn test_max_distance_boundaries() {
    let x = vec![0.0, 10.0, 20.0, 30.0];

    // At left boundary
    let win = Window::new(0, 2).unwrap();
    let dist = win.max_distance(&x, 0.0);
    assert_eq!(dist, 20.0, "Distance from left edge");

    // At right boundary
    let win = Window::new(1, 3).unwrap();
    let dist = win.max_distance(&x, 30.0);
    assert_eq!(dist, 20.0, "Distance from right edge");
}

/// Test window initialization with n=1.
#[test]
fn test_initialize_window_n_equals_one() {
    let win = Window::initialize(0, 1, 1);
    assert_eq!(win.left, 0);
    assert_eq!(win.right, 0);
    assert_eq!(win.len(), 1);
}

/// Test window initialization with window_size > n.
#[test]
fn test_initialize_window_size_exceeds_n() {
    let n = 5;
    let window_size = 100;

    for idx in 0..n {
        let win = Window::initialize(idx, window_size, n);
        assert_eq!(win.left, 0, "Should start at 0");
        assert_eq!(win.right, n - 1, "Should end at n-1");
        assert_eq!(win.len(), n, "Should span entire array");
    }
}

/// Test window recenter with uniform spacing.
#[test]
fn test_recenter_uniform_spacing() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let n = x.len();

    let mut win = Window::new(0, 2).unwrap();

    // Recenter for each point
    for current in 0..n {
        win.recenter(&x, current, n);
        assert!(
            win.left <= current && current <= win.right,
            "Current point should be in window"
        );
    }
}

/// Test window recenter with clustered data.
#[test]
fn test_recenter_clustered_data() {
    // Two clusters: [0, 1, 2] and [100, 101, 102]
    let x = vec![0.0, 1.0, 2.0, 100.0, 101.0, 102.0];
    let n = x.len();

    let mut win = Window::new(0, 2).unwrap();

    // When we reach the second cluster, window should shift
    win.recenter(&x, 3, n); // At 100.0

    // Window should prefer the closer cluster
    assert!(win.left >= 3, "Window should shift to second cluster");
}

/// Test window is_empty.
#[test]
fn test_window_is_empty() {
    let win = Window::new(5, 5).unwrap();
    assert!(!win.is_empty(), "Single-point window should not be empty");

    let win = Window::new(0, 10).unwrap();
    assert!(!win.is_empty(), "Normal window should not be empty");
}

/// Test window recenter doesn't go out of bounds.
#[test]
fn test_recenter_stays_in_bounds() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let n = x.len();

    let mut win = Window::new(0, 2).unwrap();

    // Recenter many times
    for _ in 0..100 {
        for current in 0..n {
            win.recenter(&x, current, n);
            assert!(win.left < n, "Left should stay in bounds");
            assert!(win.right < n, "Right should stay in bounds");
            assert!(win.left <= win.right, "Left should be <= right");
        }
    }
}
