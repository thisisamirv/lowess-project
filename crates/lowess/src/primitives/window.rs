//! Windowing primitives for LOWESS smoothing.
//!
//! This module provides low-level data structures for managing sliding windows
//! over sorted datasets, ensuring each local regression uses the nearest neighbors.

// External dependencies
use num_traits::Float;

// Inclusive window bounds `[left, right]` for a local fit.
#[derive(Copy, Clone, Debug)]
pub struct Window {
    // Left boundary index (inclusive).
    pub left: usize,

    // Right boundary index (inclusive).
    pub right: usize,
}

impl Window {
    // Initialize window boundaries for the first point in a sequence.
    #[inline]
    pub fn initialize(idx: usize, window_size: usize, n: usize) -> Self {
        debug_assert!(
            window_size >= 1,
            "initialize_window: window_size must be at least 1"
        );

        if window_size >= n {
            return Self {
                left: 0,
                right: n.saturating_sub(1),
            };
        }

        let half = window_size / 2;
        let mut left = idx.saturating_sub(half);
        let max_left = n - window_size;
        if left > max_left {
            left = max_left;
        }

        let right = left + window_size - 1;
        Self { left, right }
    }

    // Update boundaries to maintain nearest-neighbor centering.
    #[inline]
    pub fn recenter<T: Float>(&mut self, x: &[T], current: usize, n: usize) {
        debug_assert!(current < n, "recenter: current index out of bounds");

        self.left = self.left.min(n - 1);
        self.right = self.right.min(n - 1);

        let x_current = x[current];

        // Search for the optimal window position (nearest neighbors)
        // Slide right: if the point after the window is closer than the leftmost point
        while self.right < n - 1 {
            let d_left = x_current - x[self.left];
            let d_right = x[self.right + 1] - x_current;

            if d_left <= d_right {
                break;
            }

            self.left += 1;
            self.right += 1;
        }

        // Slide left: if the point before the window is closer or as close as the rightmost point
        while self.left > 0 {
            let d_left = x_current - x[self.left - 1];
            let d_right = x[self.right] - x_current;

            if d_right <= d_left {
                break;
            }

            self.left -= 1;
            self.right -= 1;
        }
    }

    // Compute the maximum distance from `x_current` to any point in the window.
    #[inline]
    pub fn max_distance<T: Float>(&self, x: &[T], x_current: T) -> T {
        T::max(x_current - x[self.left], x[self.right] - x_current)
    }

    // Calculate window size q from fraction alpha and data length n.
    #[inline]
    pub fn calculate_span<T: Float>(n: usize, frac: T) -> usize {
        let epsilon = T::from(1e-5).unwrap();
        let frac_n = frac * T::from(n).unwrap() + epsilon;
        let frac_n_int = frac_n.to_usize().unwrap_or(0);
        usize::max(2, usize::min(n, frac_n_int))
    }

    // Check if the window is empty.
    #[allow(dead_code)]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // Get the number of points in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.right - self.left + 1
    }
}
