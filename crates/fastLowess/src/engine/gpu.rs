//! GPU-accelerated execution engine for LOWESS smoothing.
//!
//! This module provides the GPU-accelerated smoothing function for LOWESS
//! operations. It leverages `wgpu` to execute local regression fits in parallel
//! on the GPU, providing maximum throughput for large-scale data processing.

// External dependencies
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use futures_intrusive::channel::shared::oneshot_channel;
use num_traits::Float;
use pollster::block_on;
use std::fmt::Debug;
use std::mem::size_of;
use std::sync::Mutex;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, Instance, InstanceDescriptor, MapMode,
    PipelineLayoutDescriptor, PollType, Queue, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

// Export dependencies from lowess crate
use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::api::LowessError;
use lowess::internals::engine::executor::{IterationResult, LowessConfig};
use lowess::internals::evaluation::cv::CVKind;
use lowess::internals::math::boundary::BoundaryPolicy;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::math::scaling::ScalingMethod;
use lowess::internals::primitives::buffer::CVBuffer;
use std::cmp::Ordering::Equal;

// Shader Source (WGSL)
const SHADER_SOURCE: &str = r#"
struct Config {
    n: u32,
    window_size: u32,
    weight_function: u32,
    zero_weight_fallback: u32, // Unused
    fraction: f32,
    delta: f32,
    median_threshold: f32,
    median_center: f32,
    is_absolute: u32,
    boundary_policy: u32,
    pad_len: u32,
    orig_n: u32,
    max_iterations: u32,
    tolerance: f32,
}

const BOUNDARY_EXTEND: u32 = 0u;
const BOUNDARY_REFLECT: u32 = 1u;
const BOUNDARY_ZERO: u32 = 2u;
const BOUNDARY_NONE: u32 = 3u;

const KERNEL_COSINE: u32 = 0u;
const KERNEL_EPANECHNIKOV: u32 = 1u;
const KERNEL_GAUSSIAN: u32 = 2u;
const KERNEL_BIWEIGHT: u32 = 3u;
const KERNEL_TRIANGLE: u32 = 4u;
const KERNEL_TRICUBE: u32 = 5u;
const KERNEL_UNIFORM: u32 = 6u;

const FALLBACK_USE_LOCAL_MEAN: u32 = 0u;
const FALLBACK_RETURN_ORIGINAL: u32 = 1u;
const FALLBACK_RETURN_NONE: u32 = 2u;
const FALLBACK_USE_LOCAL_MEDIAN: u32 = 2u;

const ROBUSTNESS_BISQUARE: u32 = 0u;
const ROBUSTNESS_HUBER: u32 = 1u;
const ROBUSTNESS_TALWAR: u32 = 2u;

const SCALING_MAD: u32 = 0u;
const SCALING_MAR: u32 = 1u;
const SCALING_MEAN: u32 = 2u;

const MODE_UPDATE_SCALE: u32 = 0u;
const MODE_UPDATE_CENTER: u32 = 1u;

struct WeightConfig {
    n: u32,
    scale: f32,
    robustness_method: u32,
    scaling_method: u32,
    median_center: f32,
    mean_abs: f32,
    anchor_count: u32,
    radix_pass: u32,
    converged: u32,
    iteration: u32,
    update_mode: u32,
}

// Group 0: Constants & Input Data
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> anchor_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> anchor_output: array<f32>;
@group(0) @binding(5) var<storage, read_write> indirect_args: array<u32>; // [x, y, z] dispatch args

// Group 1: Topology
@group(1) @binding(0) var<storage, read_write> interval_map: array<u32>;

// Group 2: State (Weights, Output, Residuals)
@group(2) @binding(0) var<storage, read_write> robustness_weights: array<f32>;
@group(2) @binding(1) var<storage, read_write> y_smooth: array<f32>;
@group(2) @binding(2) var<storage, read_write> residuals: array<f32>;
@group(2) @binding(3) var<storage, read_write> y_prev: array<f32>;

// Group 3: Aux (Reduction & Weight Config)
@group(3) @binding(0) var<storage, read_write> w_config: WeightConfig;
@group(3) @binding(1) var<storage, read_write> reduction: array<f32>;
@group(3) @binding(2) var<storage, read_write> std_errors: array<f32>;
@group(3) @binding(3) var<storage, read_write> global_histogram: array<atomic<u32>, 256>;
@group(3) @binding(4) var<storage, read_write> global_max_diff: atomic<u32>;
@group(3) @binding(5) var<storage, read_write> scan_block_sums: array<u32>;
@group(3) @binding(6) var<storage, read_write> scan_indices: array<u32>;

// Workgroup shared memory for scan
var<workgroup> s_scan: array<u32, 256>;

// -----------------------------------------------------------------------------
// Kernel: Loop Control & Dispatch
// -----------------------------------------------------------------------------

@compute @workgroup_size(1)
fn init_loop_state() {
    w_config.converged = 0u;
    w_config.iteration = 0u;
    w_config.radix_pass = 0u;
}

@compute @workgroup_size(256)
fn clear_histogram(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x < 256u) {
        atomicStore(&global_histogram[global_id.x], 0u);
    }
}

@compute @workgroup_size(1)
fn reset_radix_pass() {
    w_config.radix_pass = 0u;
}

@compute @workgroup_size(1)
fn inc_radix_pass() {
    w_config.radix_pass = w_config.radix_pass + 1u;
}


@compute @workgroup_size(1)
fn set_mode_scale() {
    w_config.update_mode = MODE_UPDATE_SCALE;
}

@compute @workgroup_size(1)
fn set_mode_center() {
    w_config.update_mode = MODE_UPDATE_CENTER;
}

@compute @workgroup_size(1)
fn finalize_convergence() {
    // Only check if iteration > 0
    if (w_config.iteration > 0u) {
        // Read global max difference (atomic)
        // Since differences are always positive, bitcast preserves ordering for max check
        let max_diff_bits = atomicLoad(&global_max_diff);
        let max_diff = bitcast<f32>(max_diff_bits);
        
        if (max_diff < config.tolerance) {
            w_config.converged = 1u;
        }
    }
}

@compute @workgroup_size(1)
fn prepare_next_pass() {
    w_config.iteration = w_config.iteration + 1u;
    // Reset global max diff for next pass
    atomicStore(&global_max_diff, 0u);
    
    let is_converged = w_config.converged == 1u;
    
    // Slot 0: fit_anchors (anchor_count threads)
    let anchor_count = w_config.anchor_count;
    if (is_converged) {
        indirect_args[0] = 0u;
    } else {
        indirect_args[0] = (anchor_count + 63u) / 64u;
    }
    indirect_args[1] = 1u;
    indirect_args[2] = 1u;
    
    // Slot 3: Standard N-thread Kernels (interpolate, etc)
    if (is_converged) {
        indirect_args[3] = 0u;
    } else {
        indirect_args[3] = (config.n + 63u) / 64u;
    }
    indirect_args[4] = 1u;
    indirect_args[5] = 1u;

    // Slot 6: Reduction Kernels (256 threads)
    if (is_converged) {
        indirect_args[6] = 0u;
    } else {
        indirect_args[6] = (config.n + 255u) / 256u;
    }
    indirect_args[7] = 1u;
    indirect_args[8] = 1u;
}

// -----------------------------------------------------------------------------
// Kernel: Update Scale Config
// Copies median result from reduction buffer to w_config
// Mode 0: Update Scale
// Mode 1: Update Median Center
// -----------------------------------------------------------------------------
@compute @workgroup_size(1)
fn update_scale_config() {
    if (w_config.converged == 1u) { return; }
    
    // Median result (from radix select)
    let median_abs = reduction[1048575u];
    
    if (w_config.update_mode == MODE_UPDATE_SCALE) {
        // Robustness fallback logic (matches RobustnessMethod::compute_scale)
        let mean_abs = w_config.mean_abs;
        
        let scale_threshold_const: f32 = 1e-7;
        let min_tuned_scale_const: f32 = 1e-12;
        let threshold = max(scale_threshold_const * mean_abs, min_tuned_scale_const);
        
        if (median_abs <= threshold) {
            w_config.scale = max(mean_abs, median_abs);
        } else {
            w_config.scale = median_abs;
        }
        
        // Ensure scale is at least MIN_TUNED_SCALE
        w_config.scale = max(w_config.scale, min_tuned_scale_const);
    } else {
        w_config.median_center = median_abs;
    }
}

// -----------------------------------------------------------------------------
// Kernel 1: Fit at Anchors
// Dispatched with num_anchors threads
// -----------------------------------------------------------------------------
var<workgroup> s_x: array<f32, 256>;
var<workgroup> s_y: array<f32, 256>;
var<workgroup> s_w: array<f32, 256>;
var<workgroup> wg_min_left: u32;
var<workgroup> wg_max_right: u32;

@compute @workgroup_size(64)
fn fit_anchors(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let anchor_id = global_id.x;
    let lid = local_id.x;
    let n = config.n;
    let window_size = config.window_size;
    let num_anchors_explicit = w_config.anchor_count;
    let num_anchors = arrayLength(&anchor_indices);

    var left = 0u;
    var right = 0u;
    var x_i = 0.0;
    var i = 0u;
    var valid_anchor = false;

    // Use explicit count from reduction buffer for bounds check
    // since indirect dispatch rounds up workgroups
    if (anchor_id < num_anchors_explicit) {
        i = anchor_indices[anchor_id];
        x_i = x[i];
        
        if (i > window_size / 2u) {
            left = i - window_size / 2u;
        }
        if (left + window_size > n) {
            if (n > window_size) {
                left = n - window_size;
            } else {
                left = 0u;
            }
        }
        right = left + window_size - 1u;
        
        // Recenter window based on coordinates (nearest neighbors)
        for (var idx_r = 0u; idx_r < 100u; idx_r++) {
            if (right >= n - 1u) { break; }
            let d_left = abs(x_i - x[left]);
            let d_right = abs(x[right + 1u] - x_i);
            if (d_left <= d_right) { break; }
            left = left + 1u;
            right = right + 1u;
        }
        for (var idx_l = 0u; idx_l < 100u; idx_l++) {
            if (left == 0u) { break; }
            let d_left = abs(x_i - x[left - 1u]);
            let d_right = abs(x[right] - x_i);
            if (d_right <= d_left) { break; }
            left = left - 1u;
            right = right - 1u;
        }
        valid_anchor = true;
    }

    var d_max = 0.0;
    if (valid_anchor) {
        d_max = max(abs(x_i - x[left]), abs(x_i - x[right]));
        if (d_max <= 0.0) {
            anchor_output[anchor_id] = y[i];
            valid_anchor = false;
        }
    }

    if (valid_anchor) {
        var sum_w = 0.0;
        var sum_wx = 0.0;
        var sum_wxx = 0.0;
        var sum_wy = 0.0;
        var sum_wxy = 0.0;
        var sum_y_window = 0.0;

        let d_max_val = max(d_max, 1e-12);
        let h1 = 0.001 * d_max_val;
        let h9 = 0.999 * d_max_val;

        for (var k = left; k <= right; k++) {
            let xj = x[k];
            let yj = y[k];
            let rw = robustness_weights[k];
            
            let dist = abs(xj - x_i);
            sum_y_window += yj;
            
            if (dist <= h9) {
                var kernel_w = 1.0;
                if (dist > h1) {
                    let u = dist / d_max_val;
                    let u2 = u * u;
                    switch (config.weight_function) {
                        case KERNEL_COSINE: { kernel_w = cos(u * 1.57079632679); break; }
                        case KERNEL_EPANECHNIKOV: { kernel_w = (1.0 - u2); break; }
                        case KERNEL_GAUSSIAN: { kernel_w = exp(-0.5 * u2); break; }
                        case KERNEL_BIWEIGHT: { let v = 1.0 - u2; kernel_w = v * v; break; }
                        case KERNEL_TRIANGLE: { kernel_w = (1.0 - u); break; }
                        case KERNEL_TRICUBE: { 
                            let v = 1.0 - u * u2; 
                            kernel_w = v * v * v; 
                            break; 
                        }
                        case KERNEL_UNIFORM: { kernel_w = 1.0; break; }
                        default: { 
                            let v = 1.0 - u * u2; 
                            kernel_w = v * v * v; 
                            break; 
                        }
                    }
                }
                
                let combined_w = rw * kernel_w;
                let rel_x = xj - x_i;
                sum_w += combined_w;
                sum_wx += combined_w * rel_x;
                sum_wxx += combined_w * rel_x * rel_x;
                sum_wy += combined_w * yj;
                sum_wxy += combined_w * rel_x * yj;
            }
        }

        let TOL: f32 = 1e-12;
        if (sum_w < TOL) {
            switch (config.zero_weight_fallback) {
                case FALLBACK_RETURN_ORIGINAL: { anchor_output[anchor_id] = y[i]; break; }
                default: { 
                    let w_size = f32(right - left + 1u);
                    anchor_output[anchor_id] = sum_y_window / max(1.0, w_size); 
                    break;
                }
            }
        } else {
            // Corrected sum of squares formula for better stability (matches CPU)
            let variance = sum_wxx - (sum_wx * sum_wx) / sum_w;
            
            let abs_tol = 1e-5;
            let rel_tol = 1e-5 * d_max_val * d_max_val;
            let tol = max(abs_tol, rel_tol);

            if (variance <= tol) {
                anchor_output[anchor_id] = sum_wy / sum_w; // mean_y
            } else {
                let covariance = sum_wxy - (sum_wx * sum_wy) / sum_w;
                let slope = covariance / variance;
                let intercept = (sum_wy - slope * sum_wx) / sum_w; // mean_y - slope * mean_x
                
                if (intercept == intercept && abs(intercept) < 1e15) {
                    anchor_output[anchor_id] = intercept;
                } else {
                    anchor_output[anchor_id] = sum_wy / sum_w;
                }
            }
        }
        
        // Final sanity check
        let final_val = anchor_output[anchor_id];
        if (final_val != final_val) {
             anchor_output[anchor_id] = y[i];
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: Interpolate
// Dispatched with N threads
// -----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn interpolate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.n) { return; }

    let k = interval_map[i];
    
    // Bounds check for last segment
    let num_anchors = arrayLength(&anchor_indices);
    let idx_l_ptr = k;
    var idx_r_ptr = k + 1u;
    if (idx_r_ptr >= num_anchors) {
        idx_r_ptr = k; // Fallback to flat if at end
    }

    let idx_l = anchor_indices[idx_l_ptr];
    let idx_r = anchor_indices[idx_r_ptr];
    
    let y_l = anchor_output[idx_l_ptr];
    let y_r = anchor_output[idx_r_ptr];
    
    let x_i = x[i];
    let x_l = x[idx_l];
    let x_r = x[idx_r];

    var fitted = 0.0;
    
    if (idx_l == idx_r) {
        fitted = y_l;
    } else {
        // Linear interpolation
        let t = (x_i - x_l) / (x_r - x_l);
        fitted = y_l + (y_r - y_l) * t;
    }

    y_smooth[i] = fitted;
    residuals[i] = y[i] - fitted;
}

// -----------------------------------------------------------------------------
// Kernel 3: Update Weights
// Dispatched with N threads
// -----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn update_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.n) { return; }

    let r = residuals[i];
    let abs_r = abs(r);
    var w = 1.0;
    
    let s = w_config.scale;
    if (s <= 1e-12) {
        w = 1.0;
    } else {
        switch (w_config.robustness_method) {
            case ROBUSTNESS_BISQUARE: {
                let cmad = 6.0 * s;
                if (abs_r <= 0.001 * cmad) {
                    w = 1.0;
                } else if (abs_r <= 0.999 * cmad) {
                    let u = abs_r / cmad;
                    let v = 1.0 - u * u;
                    w = v * v;
                } else {
                    w = 0.0;
                }
            }
            case ROBUSTNESS_HUBER: {
                let c = 1.345;
                let u = abs_r / s;
                if (u <= c) {
                    w = 1.0;
                } else {
                    w = c / u;
                }
            }
            case ROBUSTNESS_TALWAR: {
                let c = 2.5;
                let u = abs_r / s;
                if (u <= c) {
                    w = 1.0;
                } else {
                    w = 0.0;
                }
            }
            default: {
                w = 1.0;
            }
        }
    }

    robustness_weights[i] = w;
}

// -----------------------------------------------------------------------------
// Kernel 4: Reductions
// -----------------------------------------------------------------------------
var<workgroup> scratch: array<f32, 512>;

@compute @workgroup_size(256)
fn reduce_sum_abs(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let i = global_id.x;
    var val = 0.0;
    if (i < config.n) {
        val = abs(residuals[i]);
    }
    
    scratch[lid.x] = val;
    workgroupBarrier();
    
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            scratch[lid.x] += scratch[lid.x + s];
        }
        workgroupBarrier();
    }
    
    if (lid.x == 0u) {
        reduction[wid.x] = scratch[0];
    }
}

@compute @workgroup_size(256)
fn reduce_max_diff(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let i = global_id.x;
    var val = 0.0;
    if (i < config.n) {
        val = abs(y_smooth[i] - y_prev[i]);
    }
    
    scratch[lid.x] = val;
    workgroupBarrier();
    
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + s]);
        }
        workgroupBarrier();
    }
    
    if (lid.x == 0u) {
        reduction[wid.x] = scratch[0];
    }
}

@compute @workgroup_size(256)
fn reduce_min_max(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let i = global_id.x;
    var val = 0.0;
    if (i < config.n) {
        let r = residuals[i];
        if (config.is_absolute != 0u) {
            val = abs(r - config.median_center);
        } else {
            val = abs(r);
        }
    } else {
        // Use sentinel values for out-of-bounds to not affect reduction
        val = 0.0; // For max
    }
    
    scratch[lid.x] = val; // Store max in first half of scratch
    // Use second half for min
    scratch[lid.x + 256u] = val;
    if (i >= config.n) {
        scratch[lid.x + 256u] = 1e30; // Sentinel for min
    }
    workgroupBarrier();
    
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + s]);
            scratch[lid.x + 256u] = min(scratch[lid.x + 256u], scratch[lid.x + 256u + s]);
        }
        workgroupBarrier();
    }
    
    if (lid.x == 0u) {
        reduction[wid.x * 2u] = scratch[0];
        reduction[wid.x * 2u + 1u] = scratch[256];
    }
}

@compute @workgroup_size(256)
fn reduce_count_below(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let i = global_id.x;
    var count = 0.0;
    if (i < config.n) {
        let r = residuals[i];
        var val = r;
        if (config.is_absolute != 0u) {
            val = abs(r - config.median_center);
        }
        
        if (val <= config.median_threshold) {
            count = 1.0;
        }
    }
    
    scratch[lid.x] = count;
    workgroupBarrier();
    
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            scratch[lid.x] += scratch[lid.x + s];
        }
        workgroupBarrier();
    }
    
    if (lid.x == 0u) {
        reduction[wid.x] = scratch[0];
    }
}

@compute @workgroup_size(1)
fn finalize_scale() {
    var total_sum = 0.0;
    let num_workgroups = (config.n + 255u) / 256u;
    for (var i = 0u; i < num_workgroups; i = i + 1u) {
        total_sum += reduction[i];
    }
    let scale_est = total_sum / f32(config.n);

    // No magic number Factor: matches CPU Mean scaling
    w_config.scale = max(scale_est, 1e-12);
}

@compute @workgroup_size(1)
fn finalize_sum() {
    let n = config.n;
    var sum = 0.0;
    let num_workgroups = (n + 255u) / 256u;
    for (var i: u32 = 0u; i < num_workgroups; i = i + 1u) {
        sum = sum + reduction[i];
    }
    w_config.mean_abs = sum / f32(max(1u, n));
    reduction[0] = sum; // Keep for compatibility if needed
}

// -----------------------------------------------------------------------------
// Kernel 5: Standard Errors
// Dispatched with N threads
// -----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn compute_se(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = config.n;
    let window_size = config.window_size;
    if (i >= n) { return; }

    let x_i = x[i];
    
    var left = 0u;
    if (i > window_size / 2u) {
        left = i - window_size / 2u;
    }
    if (left + window_size > n) {
        if (n > window_size) {
            left = n - window_size;
        } else {
            left = 0u;
        }
    }
    var right = left + window_size - 1u;

    // Recenter window based on coordinates
    for (var idx_r = 0u; idx_r < 100u; idx_r++) {
        if (right >= n - 1u) { break; }
        let d_left = abs(x_i - x[left]);
        let d_right = abs(x[right + 1u] - x_i);
        if (d_left <= d_right) { break; }
        left = left + 1u;
        right = right + 1u;
    }
    for (var idx_l = 0u; idx_l < 100u; idx_l++) {
        if (left == 0u) { break; }
        let d_left = abs(x_i - x[left - 1u]);
        let d_right = abs(x[right] - x_i);
        if (d_right <= d_left) { break; }
        left = left - 1u;
        right = right - 1u;
    }

    let d_max = max(abs(x_i - x[left]), abs(x_i - x[right]));
    if (d_max <= 1e-12) {
        std_errors[i] = 0.0;
        return;
    }

    var sum_w = 0.0;
    var sum_wr2 = 0.0;
    let d_max_val = max(d_max, 1e-9);
    for (var k = left; k <= right; k++) {
        let xj = x[k];
        let smoothed_j = y_smooth[k];
        let yj = y[k];
        let rw = robustness_weights[k];
        
        let r = yj - smoothed_j;
        let dist = abs(xj - x_i);
        let u = dist / d_max_val;
        
        if (u < 1.0) {
            var kernel_w = 1.0;
            let u2 = u * u;
            switch (config.weight_function) {
                case KERNEL_COSINE: { kernel_w = cos(u * 1.57079632679); break; }
                case KERNEL_EPANECHNIKOV: { kernel_w = (1.0 - u2); break; }
                case KERNEL_GAUSSIAN: { kernel_w = exp(-0.5 * u2); break; }
                case KERNEL_BIWEIGHT: { let v = 1.0 - u2; kernel_w = v * v; break; }
                case KERNEL_TRIANGLE: { kernel_w = (1.0 - u); break; }
                case KERNEL_TRICUBE: { let v = 1.0 - u * u2; kernel_w = v * v * v; break; }
                case KERNEL_UNIFORM: { break; }
                default: { let v = 1.0 - u * u2; kernel_w = v * v * v; break; }
            }
            
            let combined_w = rw * kernel_w;
            sum_w += combined_w;
            sum_wr2 += combined_w * r * r;
        }
    }

    let LINEAR_PARAMS = 2.0;
    if (sum_w > LINEAR_PARAMS + 1e-6) {
        let df = sum_w - LINEAR_PARAMS;
        let variance = sum_wr2 / df;
        
        // Find kernel weight for current point (distance = 0)
        var w_idx = 1.0;
        switch (config.weight_function) {
            case KERNEL_COSINE: { w_idx = 1.0; break; }
            case KERNEL_EPANECHNIKOV: { w_idx = 1.0; break; }
            case KERNEL_GAUSSIAN: { w_idx = 1.0; break; }
            case KERNEL_BIWEIGHT: { w_idx = 1.0; break; }
            case KERNEL_TRIANGLE: { w_idx = 1.0; break; }
            case KERNEL_TRICUBE: { w_idx = 1.0; break; }
            case KERNEL_UNIFORM: { w_idx = 1.0; break; }
            default: { w_idx = 1.0; break; }
        }
        w_idx = w_idx * robustness_weights[i];
        
        let leverage = w_idx / sum_w;
        std_errors[i] = sqrt(variance * leverage);
    } else {
        std_errors[i] = 0.0;
    }
}

@compute @workgroup_size(256)
fn pad_data(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let pad_len = config.pad_len;
    let orig_n = config.orig_n;
    
    if (i >= pad_len) { return; } 
    
    // Prefix padding
    let x0 = x[pad_len];
    let y0 = y[pad_len];
    let dx_pre = x[pad_len+1] - x[pad_len];
    
    switch (config.boundary_policy) {
        case BOUNDARY_EXTEND: {
            x[pad_len - 1u - i] = x0 - f32(i + 1u) * dx_pre;
            y[pad_len - 1u - i] = y0;
        }
        case BOUNDARY_REFLECT: {
            x[pad_len - 1u - i] = x0 - (x[pad_len + 1u + i] - x0);
            y[pad_len - 1u - i] = y[pad_len + 1u + i];
        }
        case BOUNDARY_ZERO: {
            x[pad_len - 1u - i] = x0 - f32(i + 1u) * dx_pre;
            y[pad_len - 1u - i] = 0.0;
        }
        default: {}
    }
    
    // Suffix padding
    let xn = x[pad_len + orig_n - 1u];
    let yn = y[pad_len + orig_n - 1u];
    let dx_suf = x[pad_len + orig_n - 1u] - x[pad_len + orig_n - 2u];
    
    let target_idx = pad_len + orig_n + i;
    if (target_idx < config.n) {
        switch (config.boundary_policy) {
            case BOUNDARY_EXTEND: {
                x[target_idx] = xn + f32(i + 1u) * dx_suf;
                y[target_idx] = yn;
            }
            case BOUNDARY_REFLECT: {
                x[target_idx] = xn + (xn - x[pad_len + orig_n - 2u - i]);
                y[target_idx] = y[pad_len + orig_n - 2u - i];
            }
            case BOUNDARY_ZERO: {
                x[target_idx] = xn + f32(i + 1u) * dx_suf;
                y[target_idx] = 0.0;
            }
            default: {}
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel 8: Initialize Weights
// Dispatched with N/256 workgroups, sets all weights to 1.0
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn init_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < config.n) {
        robustness_weights[i] = 1.0;
    }
}

// -----------------------------------------------------------------------------
// Kernel 9: Mark Anchor Candidates
// Marks points where x[i] - x[i-1] > delta (potential anchor positions)
// Uses anchor_output as temporary storage for candidate flags (0 or 1)
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn mark_anchor_candidates(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = config.n;
    
    if (i >= n) {
        return;
    }
    
    var flag = 0u;
    
    if (i == 0u) {
        flag = 1u;
    } else if (i == n - 1u) {
        flag = 1u;
    } else if (config.delta < 1e-10) {
        // Force all points as anchors if delta is effectively zero
        flag = 1u;
    } else {
        // Binning strategy: floor(x / delta)
        let delta = config.delta;
        let x_curr = x[i];
        let x_prev = x[i - 1u];
        
        // Check if bin index changed
        let bin_curr = floor(x_curr / delta);
        let bin_prev = floor(x_prev / delta);
        
        if (bin_curr != bin_prev) {
            flag = 1u;
        }
    }
    
    scan_indices[i] = flag;
}

// -----------------------------------------------------------------------------
// Kernel: Block Scan (Prefix Sum within workgroup)
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn scan_block(@builtin(global_invocation_id) global_id: vec3<u32>, 
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let i = global_id.x;
    let lid = local_id.x;
    let wid = workgroup_id.x;
    
    var val = 0u;
    if (i < config.n) {
        val = scan_indices[i];
    }
    
    // Load into shared memory
    s_scan[lid] = val;
    workgroupBarrier();
    
    // Hillis-Steele Scan (Inclusive)
    for (var s = 1u; s < 256u; s = s * 2u) {
        var temp = 0u;
        if (lid >= s) {
            temp = s_scan[lid - s];
        }
        workgroupBarrier();
        if (lid >= s) {
            s_scan[lid] = s_scan[lid] + temp;
        }
        workgroupBarrier();
    }
    
    val = s_scan[lid];
    
    if (i < config.n) {
        scan_indices[i] = val; // Write inclusive sum temporarily
    }
    
    // Write total block sum to auxiliary
    if (lid == 255u) {
        scan_block_sums[wid] = val;
    }
}

// -----------------------------------------------------------------------------
// Kernel: Scan Aux Serial
// Scans the block sums (up to 65k blocks supported by single thread, though inefficient)
// -----------------------------------------------------------------------------
@compute @workgroup_size(1)
fn scan_aux_serial() {
    let num_blocks = (config.n + 255u) / 256u;
    
    // In-place exclusive scan of block sums
    // We need exclusive scan so that block i gets sum of 0..i-1
    var sum = 0u;
    for (var i = 0u; i < num_blocks; i = i + 1u) {
        let val = scan_block_sums[i];
        scan_block_sums[i] = sum;
        sum = sum + val;
    }
}

// -----------------------------------------------------------------------------
// Kernel: Scan Add Base
// Adds base offset (from scanned aux buffer) to each block
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn scan_add_base(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let i = global_id.x;
    let wid = workgroup_id.x;
    
    if (i >= config.n) { return; }
    if (wid > 0u) { // Only add base if not the first block
        let base_val = scan_block_sums[wid]; // Exclusive scan means index 'wid' has sum of 0..wid-1
        scan_indices[i] = scan_indices[i] + base_val;
    }
}

// -----------------------------------------------------------------------------
// Kernel: Compact Anchors
// Reads original flags (re-computed) and scan result to scatter indices
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn compact_anchors(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = config.n;
    if (i >= n) { return; }
    
    // Re-compute flag to know if we write
    var flag = false;
    
    if (i == 0u || i == n - 1u) {
        flag = true;
    } else if (config.delta < 1e-10) {
        flag = true;
    } else {
        let delta = config.delta;
        let bin_curr = floor(x[i] / delta);
        let bin_prev = floor(x[i - 1u] / delta);
        if (bin_curr != bin_prev) {
            flag = true;
        }
    }
    
    if (flag) {
        // scan_indices[i] contains inclusive prefix sum.
        // The index in anchor_indices is (sum - 1).
        let write_idx = scan_indices[i] - 1u;
        anchor_indices[write_idx] = i;
    }
    
    // define max anchor count
    if (i == n - 1u) {
        // Last element's scan value is the total count
        let total_anchors = scan_indices[i];
        w_config.anchor_count = total_anchors;
    }
}

// -----------------------------------------------------------------------------
// Kernel 10: Select Anchors (Greedy)
// Single workgroup kernel that performs greedy anchor selection.
// Reads candidate flags from anchor_output, writes final anchors to anchor_indices.
// Writes anchor count to reduction[0].
// -----------------------------------------------------------------------------
@compute @workgroup_size(1)
fn select_anchors_greedy() {
    let n = config.n;
    
    var anchor_count: u32 = 0u;
    var last_anchor_x: f32 = x[0];
    
    // First point is always an anchor
    anchor_indices[anchor_count] = 0u;
    anchor_count = 1u;
    
    // Greedy selection: select anchor if distance from last anchor > delta
    for (var i: u32 = 1u; i < n; i = i + 1u) {
        let curr_x = x[i];
        
        if (curr_x - last_anchor_x > config.delta) {
            anchor_indices[anchor_count] = i;
            anchor_count = anchor_count + 1u;
            last_anchor_x = curr_x;
        }
    }
    
    // Ensure last point is an anchor
    let last_idx = n - 1u;
    if (anchor_indices[anchor_count - 1u] != last_idx) {
        anchor_indices[anchor_count] = last_idx;
        anchor_count = anchor_count + 1u;
    }
    
    // Store anchor count in w_config (persistent) instead of reduction (volatile)
    w_config.anchor_count = anchor_count;
    // Also store in reduction[0] for legacy/compatibility if needed, but we switched consumers
    reduction[0] = f32(anchor_count);
}

// -----------------------------------------------------------------------------
// Kernel 11: Compute Intervals
// Assigns each point to its anchor interval based on selected anchors.
// Reads anchor count from reduction[0].
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn compute_intervals(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = config.n;
    let anchor_count = w_config.anchor_count;
    
    if (i >= n) {
        return;
    }
    
    // Binary search for correct interval
    var left: u32 = 0u;
    var right: u32 = anchor_count - 1u;
    
    while (left < right) {
        let mid = (left + right + 1u) / 2u;
        if (anchor_indices[mid] <= i) {
            left = mid;
        } else {
            right = mid - 1u;
        }
    }
    
    interval_map[i] = left;
}

// -----------------------------------------------------------------------------
// Radix Sort for Median Computation
// Uses 8-bit digits (4 passes for 32 bits)
// -----------------------------------------------------------------------------

// Convert f32 to sortable u32 (handles sign bit for correct ordering)
fn f32_to_sortable_u32(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    // If sign bit is set (negative), flip all bits; else flip just sign bit
    let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
    return bits ^ mask;
}

fn sortable_u32_to_f32(u: u32) -> f32 {
    // Reverse the transformation
    let mask = select(0x80000000u, 0xFFFFFFFFu, (u & 0x80000000u) == 0u);
    return bitcast<f32>(u ^ mask);
}

// Shared histogram for radix counting (256 bins for 8-bit digit)
var<workgroup> local_histogram: array<atomic<u32>, 256>;

// Prepare residuals for MAR: reduction[i] = abs(residuals[i])
@compute @workgroup_size(256)
fn prepare_mar_residuals(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = config.n;
    if (i < n) {
        reduction[i] = abs(residuals[i]);
    }
}

// Prepare residuals for MAD Step 1: reduction[i] = residuals[i] (signed)
@compute @workgroup_size(256)
fn prepare_residuals_signed(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = config.n;
    if (i < n) {
        reduction[i] = residuals[i];
    }
}

// Prepare residuals for MAD Step 2: reduction[i] = abs(residuals[i] - w_config.median_center)
@compute @workgroup_size(256)
fn prepare_mad_residuals(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = config.n;
    if (i < n) {
        reduction[i] = abs(residuals[i] - w_config.median_center);
    }
}

// Radix histogram: count occurrences of each 8-bit digit
// Uses reduction[0..n] as source
@compute @workgroup_size(256)
fn radix_histogram(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>) {
    let n = config.n;
    let radix_pass = w_config.radix_pass; // 0, 1, 2, or 3
    let shift = radix_pass * 8u;
    
    // Initialize local histogram
    atomicStore(&local_histogram[local_id.x], 0u);
    workgroupBarrier();
    
    // Count digits
    let i = global_id.x;
    if (i < n) {
        let val = f32_to_sortable_u32(reduction[i]);
        let digit = (val >> shift) & 0xFFu;
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();
    
    // Add to global histogram atomics
    if (local_id.x < 256u) {
        let count = atomicLoad(&local_histogram[local_id.x]);
        if (count > 0u) {
            atomicAdd(&global_histogram[local_id.x], count);
        }
    }
}

// Prefix sum on histogram (single workgroup, 256 elements)
@compute @workgroup_size(1)
fn radix_prefix_sum() {
    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        let count = atomicLoad(&global_histogram[i]);
        atomicStore(&global_histogram[i], sum);
        sum = sum + count;
    }
}

// Scatter elements to sorted positions
// Reads from reduction[0..n], writes to reduction[524288..]
// We assume n <= 524288 (half of the 1M element reduction buffer)
// NOTE: Single-threaded to allow stable sort (sequential scatter)
@compute @workgroup_size(1)
fn radix_scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = config.n;
    if (n > 524288u) { return; } // Safety check for buffer paging
    
    let radix_pass = w_config.radix_pass;
    let shift = radix_pass * 8u;
    
    // Sequential loop over all elements to preserve stability
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let element = reduction[i];
        let sortable_val = f32_to_sortable_u32(element);
        let digit = (sortable_val >> shift) & 0xffu;
        
        // Use atomicAdd on global_histogram to get unique global position
        // Since we are single-threaded, this is deterministic and stable
        let pos = atomicAdd(&global_histogram[digit], 1u);
        reduction[524288u + pos] = element;
    }
}

// Copy back from upper half to lower half
@compute @workgroup_size(256)
fn radix_copy_back(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < config.n) {
        reduction[i] = reduction[524288u + i];
    }
}

// Select median from sorted reduction
// Result stored in reduction[0] (compact for download)
@compute @workgroup_size(1)
fn select_median() {
    let n = config.n;
    if (n == 0u) { return; }
    let mid = n / 2u;
    
    var res = 0.0;
    if (n % 2u == 0u) {
        res = (reduction[mid - 1u] + reduction[mid]) * 0.5;
    } else {
        res = reduction[mid];
    }
    reduction[1048575u] = res;
}

// -----------------------------------------------------------------------------
// Input Sort Kernels (x, y)
// -----------------------------------------------------------------------------

// Histogram for input sorting: count occurences of each 8-bit digit of x
@compute @workgroup_size(256)
fn sort_x_histogram(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let n = config.orig_n; // Only sort valid data
    let pad = config.pad_len;
    let radix_pass = w_config.radix_pass;
    let shift = radix_pass * 8u;

    atomicStore(&local_histogram[local_id.x], 0u);
    workgroupBarrier();

    let i = global_id.x;
    if (i < n) {
        let val = f32_to_sortable_u32(x[i + pad]);
        let digit = (val >> shift) & 0xFFu;
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();

    if (local_id.x < 256u) {
        let count = atomicLoad(&local_histogram[local_id.x]);
        if (count > 0u) {
            atomicAdd(&global_histogram[local_id.x], count);
        }
    }
}

// Scatter x and y to temp buffers (y_smooth as x_tmp, residuals as y_tmp)
// NOTE: Single-threaded for stability
@compute @workgroup_size(1)
fn sort_x_scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = config.orig_n; // Only sort valid data
    let pad = config.pad_len;
    let radix_pass = w_config.radix_pass;
    let shift = radix_pass * 8u;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let val_x = x[i + pad];
        let val_y = y[i + pad];
        let sortable_val = f32_to_sortable_u32(val_x);
        let digit = (sortable_val >> shift) & 0xffu;

        let pos = atomicAdd(&global_histogram[digit], 1u);
        
        // Write to temp buffers with offset
        y_smooth[pos + pad] = val_x;
        residuals[pos + pad] = val_y;
    }
}

// Copy back from temp buffers to x, y
@compute @workgroup_size(256)
fn sort_x_copy_back(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < config.orig_n) {
        let pad = config.pad_len;
        let idx = i + pad;
        x[idx] = y_smooth[idx];
        y[idx] = residuals[idx];
    }
}

"#;

const ROBUSTNESS_BISQUARE: u32 = 0;
const ROBUSTNESS_HUBER: u32 = 1;
const ROBUSTNESS_TALWAR: u32 = 2;

const SCALING_MAD: u32 = 0;
const SCALING_MAR: u32 = 1;
const SCALING_MEAN: u32 = 2;

pub static GLOBAL_EXECUTOR: Mutex<Option<GpuExecutor>> = Mutex::new(None);

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuConfig {
    pub n: u32,
    pub window_size: u32,
    pub weight_function: u32,
    pub zero_weight_fallback: u32,
    pub fraction: f32,
    pub delta: f32,
    pub median_threshold: f32,
    pub median_center: f32,
    pub is_absolute: u32,
    pub boundary_policy: u32,
    pub pad_len: u32,
    pub orig_n: u32,
    pub max_iterations: u32,
    pub tolerance: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct WeightConfig {
    pub n: u32,
    pub scale: f32,
    pub robustness_method: u32,
    pub scaling_method: u32,
    pub median_center: f32,
    pub mean_abs: f32,
    pub anchor_count: u32,
    pub radix_pass: u32,
    pub converged: u32,
    pub iteration: u32,
    pub update_mode: u32,
}

pub struct GpuExecutor {
    device: Device,
    pub queue: Queue,

    // Pipelines
    fit_pipeline: ComputePipeline,
    interpolate_pipeline: ComputePipeline,
    weight_pipeline: ComputePipeline,
    reduce_max_diff_pipeline: ComputePipeline,
    reduce_sum_abs_pipeline: ComputePipeline,
    finalize_scale_pipeline: ComputePipeline,
    init_weights_pipeline: ComputePipeline,
    compute_intervals_pipeline: ComputePipeline,
    prepare_residuals_signed_pipeline: ComputePipeline,
    prepare_mar_residuals_pipeline: ComputePipeline,
    prepare_mad_residuals_pipeline: ComputePipeline,
    radix_histogram_pipeline: ComputePipeline,
    radix_prefix_sum_pipeline: ComputePipeline,
    radix_scatter_pipeline: ComputePipeline,
    radix_copy_back_pipeline: ComputePipeline,
    select_median_pipeline: ComputePipeline,
    sort_x_histogram_pipeline: ComputePipeline,
    sort_x_scatter_pipeline: ComputePipeline,
    sort_x_copy_back_pipeline: ComputePipeline,
    se_pipeline: ComputePipeline,
    pad_pipeline: ComputePipeline,
    update_scale_config_pipeline: ComputePipeline,
    init_loop_state_pipeline: ComputePipeline,
    finalize_convergence_pipeline: ComputePipeline,
    prepare_next_pass_pipeline: ComputePipeline,
    clear_histogram_pipeline: ComputePipeline,
    reset_radix_pass_pipeline: ComputePipeline,
    inc_radix_pass_pipeline: ComputePipeline,
    set_mode_scale_pipeline: ComputePipeline,
    set_mode_center_pipeline: ComputePipeline,
    scan_block_pipeline: ComputePipeline,
    scan_add_base_pipeline: ComputePipeline,
    scan_aux_pipeline: ComputePipeline,
    compact_anchors_pipeline: ComputePipeline,
    mark_anchor_candidates_pipeline: ComputePipeline,
    finalize_sum_pipeline: ComputePipeline,

    // Buffers - Group 0
    config_buffer: Option<Buffer>,
    x_buffer: Option<Buffer>,
    y_buffer: Option<Buffer>,
    anchor_indices_buffer: Option<Buffer>,
    anchor_output_buffer: Option<Buffer>,
    indirect_buffer: Option<Buffer>,

    // Buffers - Group 1
    interval_map_buffer: Option<Buffer>,

    // Buffers - Group 2
    weights_buffer: Option<Buffer>,
    y_smooth_buffer: Option<Buffer>,
    y_prev_buffer: Option<Buffer>,
    residuals_buffer: Option<Buffer>,
    histogram_buffer: Option<Buffer>,

    // Buffers - Group 3
    w_config_buffer: Option<Buffer>,
    pub reduction_buffer: Option<Buffer>,
    std_errors_buffer: Option<Buffer>,
    global_max_diff_buffer: Option<Buffer>,
    median_buffer: Option<Buffer>,
    scan_block_sums_buffer: Option<Buffer>,
    scan_indices_buffer: Option<Buffer>,

    // Staging
    staging_buffer: Option<Buffer>,

    // Bind Groups
    bg0_data: Option<BindGroup>,
    bg1_topo: Option<BindGroup>,
    bg2_state: Option<BindGroup>,
    bg3_aux: Option<BindGroup>,
    bg3_median: Option<BindGroup>, // Alternate BG3 with median_buffer as target

    n: u32,
    orig_n: u32,
    num_anchors: u32,
}
impl GpuExecutor {
    pub async fn new() -> Result<Self, String> {
        let instance = Instance::new(&InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await;

        let adapter = adapter.map_err(|_| "No GPU adapter found")?;

        let (device, queue): (Device, Queue) = adapter
            .request_device(&Default::default())
            .await
            .map_err(|e| format!("Device error: {:?}", e))?;

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("LOWESS Shader"),
            source: ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // Layouts
        let bind_group_layout_0 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BG0 Data"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group_layout_1 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BG1 Topo"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group_layout_2 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BG2 State"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group_layout_3 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BG3 Aux"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[
                &bind_group_layout_0,
                &bind_group_layout_1,
                &bind_group_layout_2,
                &bind_group_layout_3,
            ],
            ..Default::default()
        });

        let create_pipeline = |entry: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Ok(Self {
            fit_pipeline: create_pipeline("fit_anchors"),
            interpolate_pipeline: create_pipeline("interpolate"),
            weight_pipeline: create_pipeline("update_weights"),
            reduce_max_diff_pipeline: create_pipeline("reduce_max_diff"),
            reduce_sum_abs_pipeline: create_pipeline("reduce_sum_abs"),
            finalize_scale_pipeline: create_pipeline("finalize_scale"),
            finalize_sum_pipeline: create_pipeline("finalize_sum"),
            init_weights_pipeline: create_pipeline("init_weights"),
            compute_intervals_pipeline: create_pipeline("compute_intervals"),
            prepare_residuals_signed_pipeline: create_pipeline("prepare_residuals_signed"),
            prepare_mar_residuals_pipeline: create_pipeline("prepare_mar_residuals"),
            prepare_mad_residuals_pipeline: create_pipeline("prepare_mad_residuals"),
            radix_histogram_pipeline: create_pipeline("radix_histogram"),
            radix_prefix_sum_pipeline: create_pipeline("radix_prefix_sum"),
            radix_scatter_pipeline: create_pipeline("radix_scatter"),
            radix_copy_back_pipeline: create_pipeline("radix_copy_back"),
            select_median_pipeline: create_pipeline("select_median"),
            sort_x_histogram_pipeline: create_pipeline("sort_x_histogram"),
            sort_x_scatter_pipeline: create_pipeline("sort_x_scatter"),
            sort_x_copy_back_pipeline: create_pipeline("sort_x_copy_back"),
            se_pipeline: create_pipeline("compute_se"),
            pad_pipeline: create_pipeline("pad_data"),
            update_scale_config_pipeline: create_pipeline("update_scale_config"),
            init_loop_state_pipeline: create_pipeline("init_loop_state"),
            finalize_convergence_pipeline: create_pipeline("finalize_convergence"),
            prepare_next_pass_pipeline: create_pipeline("prepare_next_pass"),
            clear_histogram_pipeline: create_pipeline("clear_histogram"),
            reset_radix_pass_pipeline: create_pipeline("reset_radix_pass"),
            inc_radix_pass_pipeline: create_pipeline("inc_radix_pass"),
            set_mode_scale_pipeline: create_pipeline("set_mode_scale"),
            set_mode_center_pipeline: create_pipeline("set_mode_center"),
            scan_block_pipeline: create_pipeline("scan_block"),
            scan_aux_pipeline: create_pipeline("scan_aux_serial"),
            scan_add_base_pipeline: create_pipeline("scan_add_base"),
            compact_anchors_pipeline: create_pipeline("compact_anchors"),
            mark_anchor_candidates_pipeline: create_pipeline("mark_anchor_candidates"),
            device,
            queue,
            config_buffer: None,
            x_buffer: None,
            y_buffer: None,
            anchor_indices_buffer: None,
            anchor_output_buffer: None,
            indirect_buffer: None,
            interval_map_buffer: None,
            weights_buffer: None,
            y_smooth_buffer: None,
            y_prev_buffer: None,
            residuals_buffer: None,
            histogram_buffer: None,
            w_config_buffer: None,
            reduction_buffer: None,
            std_errors_buffer: None,
            global_max_diff_buffer: None,
            median_buffer: None,
            scan_block_sums_buffer: None,
            scan_indices_buffer: None,
            staging_buffer: None,
            bg0_data: None,
            bg1_topo: None,
            bg2_state: None,
            bg3_aux: None,
            bg3_median: None,
            n: 0,
            orig_n: 0,
            num_anchors: 0,
        })
    }

    fn ensure_buffer_capacity(
        device: &Device,
        label: &str,
        buffer_opt: &mut Option<Buffer>,
        size_required: u64,
        usage: BufferUsages,
    ) -> bool {
        let mut created_new = false;
        if let Some(buffer) = buffer_opt.as_ref()
            && buffer.size() < size_required
        {
            *buffer_opt = None;
        }

        if buffer_opt.is_none() {
            *buffer_opt = Some(device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: size_required,
                usage,
                mapped_at_creation: false,
            }));
            created_new = true;
        }
        created_new
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reset_buffers(
        &mut self,
        x: &[f32],
        y: &[f32],
        config: GpuConfig,
        robustness_method: u32,
        scaling_method: u32,
    ) {
        let n_padded = config.n;
        let orig_n = config.orig_n;
        // Anchor buffer sized for worst case (every point is anchor)
        let max_anchors = n_padded + 2; // +2 for first and last
        let n_bytes_padded = (n_padded as usize * 4) as u64;
        let anchor_bytes = ((max_anchors as usize).max(256) * 4) as u64;

        let mut bg_needs_update = false;

        // Group 0: Config (Uniform)
        if Self::ensure_buffer_capacity(
            &self.device,
            "Config",
            &mut self.config_buffer,
            size_of::<GpuConfig>() as u64,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }
        self.queue.write_buffer(
            self.config_buffer.as_ref().unwrap(),
            0,
            cast_slice(&[config]),
        );

        // Group 0: X, Y, Anchors, AnchorOutput
        if Self::ensure_buffer_capacity(
            &self.device,
            "X",
            &mut self.x_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }
        self.queue.write_buffer(
            self.x_buffer.as_ref().unwrap(),
            (config.pad_len as usize * 4) as u64,
            cast_slice(x),
        );

        if Self::ensure_buffer_capacity(
            &self.device,
            "Y",
            &mut self.y_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if Self::ensure_buffer_capacity(
            &self.device,
            "IndirectArgs",
            &mut self.indirect_buffer,
            128, // Multiple slots for indirect dispatch (Fit, Std, Reduction, etc)
            BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC,
        ) {
            bg_needs_update = true;
        }
        self.queue.write_buffer(
            self.y_buffer.as_ref().unwrap(),
            (config.pad_len as usize * 4) as u64,
            cast_slice(y),
        );

        if Self::ensure_buffer_capacity(
            &self.device,
            "AnchorIndices",
            &mut self.anchor_indices_buffer,
            anchor_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if Self::ensure_buffer_capacity(
            &self.device,
            "AnchorOutput",
            &mut self.anchor_output_buffer,
            anchor_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if bg_needs_update || self.bg0_data.is_none() {
            self.bg0_data = Some(
                self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("BG0"),
                    layout: &self.fit_pipeline.get_bind_group_layout(0),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.config_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: self.x_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: self.y_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: self
                                .anchor_indices_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: self
                                .anchor_output_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: self.indirect_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                    ],
                }),
            );
        }

        // Group 1: Interval Map
        bg_needs_update = false;
        let interval_bytes = (n_padded as usize * 8) as u64;
        if Self::ensure_buffer_capacity(
            &self.device,
            "IntervalMap",
            &mut self.interval_map_buffer,
            interval_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if bg_needs_update || self.bg1_topo.is_none() {
            self.bg1_topo = Some(
                self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("BG1"),
                    layout: &self.fit_pipeline.get_bind_group_layout(1),
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: self
                            .interval_map_buffer
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    }],
                }),
            );
        }

        // Group 2: State
        bg_needs_update = false;
        if Self::ensure_buffer_capacity(
            &self.device,
            "RobustnessWeights",
            &mut self.weights_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        ) {
            bg_needs_update = true;
        }

        if Self::ensure_buffer_capacity(
            &self.device,
            "YSmooth",
            &mut self.y_smooth_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if Self::ensure_buffer_capacity(
            &self.device,
            "Residuals",
            &mut self.residuals_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if Self::ensure_buffer_capacity(
            &self.device,
            "YPrev",
            &mut self.y_prev_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if bg_needs_update || self.bg2_state.is_none() {
            self.bg2_state = Some(self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("BG2"),
                layout: &self.fit_pipeline.get_bind_group_layout(2),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.weights_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.y_smooth_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.residuals_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: self.y_prev_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                ],
            }));
        }

        // Group 3: Aux
        bg_needs_update = false;
        if Self::ensure_buffer_capacity(
            &self.device,
            "WConfig",
            &mut self.w_config_buffer,
            size_of::<WeightConfig>() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        // Reduction buffer sized for 1M elements (4MB) to support radix sort paging
        let reduction_size = ((1024 * 1024) as usize * 4) as u64;
        if Self::ensure_buffer_capacity(
            &self.device,
            "Reduction",
            &mut self.reduction_buffer,
            reduction_size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if Self::ensure_buffer_capacity(
            &self.device,
            "StdErrors",
            &mut self.std_errors_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if Self::ensure_buffer_capacity(
            &self.device,
            "Histogram",
            &mut self.histogram_buffer,
            1024, // 256 * 4
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        // Global Max Diff Buffer (Atomic u32)
        if Self::ensure_buffer_capacity(
            &self.device,
            "Global Max Diff",
            &mut self.global_max_diff_buffer,
            4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if Self::ensure_buffer_capacity(
            &self.device,
            "ScanBlockSums",
            &mut self.scan_block_sums_buffer,
            (n_padded / 256 * 4 + 4) as u64, // Enough block sums
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if Self::ensure_buffer_capacity(
            &self.device,
            "ScanIndices",
            &mut self.scan_indices_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if bg_needs_update || self.bg3_aux.is_none() {
            self.bg3_aux = Some(
                self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("BG3"),
                    layout: &self.fit_pipeline.get_bind_group_layout(3),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.w_config_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: self.reduction_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: self.std_errors_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: self.histogram_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: self
                                .global_max_diff_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: self
                                .scan_block_sums_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 6,
                            resource: self
                                .scan_indices_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                    ],
                }),
            );
        }

        if Self::ensure_buffer_capacity(
            &self.device,
            "Median",
            &mut self.median_buffer,
            reduction_size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if bg_needs_update || self.bg3_median.is_none() {
            self.bg3_median = Some(
                self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("BG3_Median"),
                    layout: &self.fit_pipeline.get_bind_group_layout(3),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.w_config_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: self.median_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: self.std_errors_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: self.histogram_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: self
                                .global_max_diff_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: self
                                .scan_block_sums_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 6,
                            resource: self
                                .scan_indices_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                    ],
                }),
            );
        }

        // Initialize WeightConfig
        self.queue.write_buffer(
            self.w_config_buffer.as_ref().unwrap(),
            0,
            bytes_of(&WeightConfig {
                n: n_padded,
                scale: 0.0,
                robustness_method,
                scaling_method,
                median_center: 0.0,
                mean_abs: 0.0,
                anchor_count: 0,
                radix_pass: 0,
                converged: 0,
                iteration: 0,
                update_mode: 0,
            }),
        );

        // Staging
        let staging_size = n_bytes_padded.max(1024); // Ensure enough space for small downloads or median
        Self::ensure_buffer_capacity(
            &self.device,
            "Staging",
            &mut self.staging_buffer,
            staging_size,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        self.n = n_padded;
        self.orig_n = orig_n;
        self.num_anchors = max_anchors; // Will be updated after anchor selection kernel runs

        // Run padding kernel if pad_len > 0
        if config.pad_len > 0 {
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Boundary Padding"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_pipeline(&self.pad_pipeline);
                pass.set_bind_group(0, self.bg0_data.as_ref().unwrap(), &[]);
                pass.set_bind_group(1, self.bg1_topo.as_ref().unwrap(), &[]);
                pass.set_bind_group(2, self.bg2_state.as_ref().unwrap(), &[]);
                pass.set_bind_group(3, self.bg3_aux.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(config.pad_len.div_ceil(64), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            let _ = self.device.poll(PollType::Wait);
        }
    }

    pub fn record_pipeline(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &ComputePipeline,
        dispatch: DispatchMode,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, self.bg0_data.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.bg1_topo.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, self.bg2_state.as_ref().unwrap(), &[]);
        pass.set_bind_group(3, self.bg3_aux.as_ref().unwrap(), &[]);
        match dispatch {
            DispatchMode::Direct(n) => pass.dispatch_workgroups(n, 1, 1),
            DispatchMode::Indirect(offset) => {
                pass.dispatch_workgroups_indirect(self.indirect_buffer.as_ref().unwrap(), offset)
            }
        }
    }

    pub fn record_pipeline_with_bg3(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &ComputePipeline,
        dispatch: DispatchMode,
        bg3: &BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, self.bg0_data.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.bg1_topo.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, self.bg2_state.as_ref().unwrap(), &[]);
        pass.set_bind_group(3, bg3, &[]);
        match dispatch {
            DispatchMode::Direct(n) => pass.dispatch_workgroups(n, 1, 1),
            DispatchMode::Indirect(offset) => {
                pass.dispatch_workgroups_indirect(self.indirect_buffer.as_ref().unwrap(), offset)
            }
        }
    }
}

pub enum DispatchMode {
    Direct(u32),
    Indirect(u64),
}

impl GpuExecutor {
    fn record_fit_pass(&self, encoder: &mut CommandEncoder) {
        // 1. Fit Anchors (Indirect Dispatch)
        self.record_pipeline(encoder, &self.fit_pipeline, DispatchMode::Indirect(0));
        // 2. Interpolate
        self.record_pipeline(
            encoder,
            &self.interpolate_pipeline,
            DispatchMode::Direct(self.n.div_ceil(64)),
        );
    }

    fn record_scale_estimation(&self, encoder: &mut CommandEncoder) {
        // 1. Sum absolute residuals
        self.record_pipeline(
            encoder,
            &self.reduce_sum_abs_pipeline,
            DispatchMode::Direct(self.n.div_ceil(256)),
        );
        // 2. Finalize scale (single thread)
        self.record_pipeline(
            encoder,
            &self.finalize_scale_pipeline,
            DispatchMode::Direct(1),
        );
    }

    fn record_update_weights(&self, encoder: &mut CommandEncoder) {
        // 3. Update Weights
        self.record_pipeline(
            encoder,
            &self.weight_pipeline,
            DispatchMode::Direct(self.n.div_ceil(64)),
        );
    }

    fn record_convergence_check(&self, encoder: &mut CommandEncoder) {
        // Reduce max differences
        self.record_pipeline(
            encoder,
            &self.reduce_max_diff_pipeline,
            DispatchMode::Direct(self.n.div_ceil(256)),
        );
    }

    pub async fn download_buffer_raw(
        &self,
        buf: &Buffer,
        size_override: Option<u64>,
        offset_override: Option<u64>,
    ) -> Option<Vec<u8>> {
        let size = size_override.unwrap_or((self.n as usize * 4) as u64);
        let offset = offset_override.unwrap_or(0);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, offset, self.staging_buffer.as_ref().unwrap(), 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.as_ref().unwrap().slice(..size);
        let (tx, rx) = oneshot_channel();
        slice.map_async(MapMode::Read, move |v| tx.send(v).unwrap());
        let _ = self.device.poll(PollType::Wait);

        if let Some(Ok(())) = rx.receive().await {
            let data = slice.get_mapped_range();
            let ret = data.to_vec();
            drop(data);
            self.staging_buffer.as_ref().unwrap().unmap();
            Some(ret)
        } else {
            None
        }
    }

    pub async fn download_buffer(
        &self,
        buf: &Buffer,
        size_override: Option<u64>,
        offset_override: Option<u64>,
    ) -> Option<Vec<f32>> {
        self.download_buffer_raw(buf, size_override, offset_override)
            .await
            .map(|raw| cast_slice::<u8, f32>(&raw).to_vec())
    }
    fn record_radix_sort_passes(
        &self,
        encoder: &mut CommandEncoder,
        bg3_override: Option<&BindGroup>,
    ) {
        // Reset radix pass to 0
        if let Some(bg3) = bg3_override {
            self.record_pipeline_with_bg3(
                encoder,
                &self.reset_radix_pass_pipeline,
                DispatchMode::Direct(1),
                bg3,
            );
        } else {
            self.record_pipeline(
                encoder,
                &self.reset_radix_pass_pipeline,
                DispatchMode::Direct(1),
            );
        }

        // Radix sort for exact median (8-bit digits, 4 passes)
        for _ in 0..4 {
            // 1. Clear Histogram
            if let Some(bg3) = bg3_override {
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.clear_histogram_pipeline,
                    DispatchMode::Direct(1),
                    bg3,
                );
            } else {
                self.record_pipeline(
                    encoder,
                    &self.clear_histogram_pipeline,
                    DispatchMode::Direct(1),
                );
            }

            // 2. Histogram
            if let Some(bg3) = bg3_override {
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.radix_histogram_pipeline,
                    DispatchMode::Indirect(24), // Slot 6 (Reduction-style, 256 threads)
                    bg3,
                );
            } else {
                self.record_pipeline(
                    encoder,
                    &self.radix_histogram_pipeline,
                    DispatchMode::Indirect(24),
                );
            }

            // 3. Prefix Sum
            if let Some(bg3) = bg3_override {
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.radix_prefix_sum_pipeline,
                    DispatchMode::Direct(1),
                    bg3,
                );
            } else {
                self.record_pipeline(
                    encoder,
                    &self.radix_prefix_sum_pipeline,
                    DispatchMode::Direct(1),
                );
            }

            // 4. Scatter
            if let Some(bg3) = bg3_override {
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.radix_scatter_pipeline,
                    DispatchMode::Direct(1),
                    bg3,
                );
            } else {
                self.record_pipeline(
                    encoder,
                    &self.radix_scatter_pipeline,
                    DispatchMode::Direct(1),
                );
            }

            // 5. Copy Back
            if let Some(bg3) = bg3_override {
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.radix_copy_back_pipeline,
                    DispatchMode::Indirect(24),
                    bg3,
                );
            } else {
                self.record_pipeline(
                    encoder,
                    &self.radix_copy_back_pipeline,
                    DispatchMode::Indirect(24),
                );
            }

            // 6. Increment Radix Pass
            if let Some(bg3) = bg3_override {
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.inc_radix_pass_pipeline,
                    DispatchMode::Direct(1),
                    bg3,
                );
            } else {
                self.record_pipeline(
                    encoder,
                    &self.inc_radix_pass_pipeline,
                    DispatchMode::Direct(1),
                );
            }
        }
    }

    #[cfg(feature = "dev")]
    pub async fn compute_median_gpu(&mut self) -> Result<f32, String> {
        // Run the 4-pass radix sort
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.record_radix_sort_passes(&mut encoder, None);
        self.queue.submit(Some(encoder.finish()));

        // Final step: Select Median
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.record_pipeline(
            &mut encoder,
            &self.select_median_pipeline,
            DispatchMode::Direct(1),
        );
        self.queue.submit(Some(encoder.finish()));

        // Download result from reduction_buffer[1048575]
        let result = self
            .download_buffer(
                self.reduction_buffer.as_ref().unwrap(),
                Some(4),
                Some(1048575 * 4),
            )
            .await
            .ok_or_else(|| "Failed to download median result".to_string())?;

        Ok(result[0])
    }

    /// Record commands to sort the input (x, y) based on x values.
    /// This uses a 4-pass radix sort and utilizes y_smooth/residuals as temporary buffers.
    fn record_sort_input(&self, encoder: &mut CommandEncoder) {
        // Reset radix pass to 0
        self.record_pipeline(
            encoder,
            &self.reset_radix_pass_pipeline,
            DispatchMode::Direct(1),
        );

        for _ in 0..4 {
            // 0. Clear Histogram
            self.record_pipeline(
                encoder,
                &self.clear_histogram_pipeline,
                DispatchMode::Direct(1),
            );

            // 1. Histogram
            self.record_pipeline(
                encoder,
                &self.sort_x_histogram_pipeline,
                DispatchMode::Direct(self.n.div_ceil(256)),
            );
            // 2. Prefix Sum
            self.record_pipeline(
                encoder,
                &self.radix_prefix_sum_pipeline,
                DispatchMode::Direct(1),
            );
            // 3. Scatter
            self.record_pipeline(
                encoder,
                &self.sort_x_scatter_pipeline,
                DispatchMode::Direct(1), // Serial scatter
            );
            // 4. Copy Back
            self.record_pipeline(
                encoder,
                &self.sort_x_copy_back_pipeline,
                DispatchMode::Direct(self.n.div_ceil(256)),
            );
            // 5. Increment Pass
            self.record_pipeline(
                encoder,
                &self.inc_radix_pass_pipeline,
                DispatchMode::Direct(1),
            );
        }
    }

    fn record_robust_scale(&self, encoder: &mut CommandEncoder, scaling_method: ScalingMethod) {
        match scaling_method {
            ScalingMethod::Mean => {
                self.record_scale_estimation(encoder);
            }
            ScalingMethod::MAR => {
                let bg3_med = self.bg3_median.as_ref().unwrap();

                // 1. Prepare Mean (for fallback logic in update_scale_config)
                self.record_pipeline(
                    encoder,
                    &self.reduce_sum_abs_pipeline,
                    DispatchMode::Direct(self.n.div_ceil(256)),
                );
                self.record_pipeline(
                    encoder,
                    &self.finalize_sum_pipeline,
                    DispatchMode::Direct(1),
                );

                // 2. Prepare absolute residuals
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.prepare_mar_residuals_pipeline,
                    DispatchMode::Direct(self.n.div_ceil(256)),
                    bg3_med,
                );

                // 3. Sort
                self.record_radix_sort_passes(encoder, Some(bg3_med));

                // 4. Select Median & Update Scale
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.select_median_pipeline,
                    DispatchMode::Direct(1),
                    bg3_med,
                );
                self.record_pipeline(
                    encoder,
                    &self.set_mode_scale_pipeline,
                    DispatchMode::Direct(1),
                );
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.update_scale_config_pipeline,
                    DispatchMode::Direct(1),
                    bg3_med,
                );
            }
            ScalingMethod::MAD => {
                let bg3_med = self.bg3_median.as_ref().unwrap();

                // 1. Prepare Mean (for fallback logic in update_scale_config)
                self.record_pipeline(
                    encoder,
                    &self.reduce_sum_abs_pipeline,
                    DispatchMode::Direct(self.n.div_ceil(256)),
                );
                self.record_pipeline(
                    encoder,
                    &self.finalize_sum_pipeline,
                    DispatchMode::Direct(1),
                );

                // 2. MAD Step 1 (Center)
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.prepare_residuals_signed_pipeline,
                    DispatchMode::Direct(self.n.div_ceil(256)),
                    bg3_med,
                );
                self.record_radix_sort_passes(encoder, Some(bg3_med));

                // 2. Select Median & Set Center
                // We transmit the median to w_config.median_center

                self.record_pipeline_with_bg3(
                    encoder,
                    &self.select_median_pipeline,
                    DispatchMode::Direct(1),
                    bg3_med,
                );
                // Switch to Center mode via kernel
                self.record_pipeline(
                    encoder,
                    &self.set_mode_center_pipeline,
                    DispatchMode::Direct(1),
                );
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.update_scale_config_pipeline,
                    DispatchMode::Direct(1),
                    bg3_med,
                );

                // 2. MAD Step 2 (MAD)
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.prepare_mad_residuals_pipeline,
                    DispatchMode::Direct(self.n.div_ceil(256)),
                    bg3_med,
                );
                self.record_radix_sort_passes(encoder, Some(bg3_med));

                // 3. Select Median & Set Scale
                // Reset mode to SCALE
                self.record_pipeline(
                    encoder,
                    &self.set_mode_scale_pipeline,
                    DispatchMode::Direct(1),
                );

                self.record_pipeline_with_bg3(
                    encoder,
                    &self.select_median_pipeline,
                    DispatchMode::Direct(1),
                    bg3_med,
                );
                self.record_pipeline_with_bg3(
                    encoder,
                    &self.update_scale_config_pipeline,
                    DispatchMode::Direct(1),
                    bg3_med,
                );
            }
        }
    }
}

// Compute standard errors for GPU results using GPU SE kernel
fn compute_intervals_gpu<T>(exec: &mut GpuExecutor, config: &LowessConfig<T>) -> Option<Vec<T>>
where
    T: Float + Debug + Send + Sync + 'static,
{
    // Check if intervals requested
    let _ = config.return_variance.as_ref()?;

    let mut encoder = exec
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Standard Error Calculation"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&exec.se_pipeline);
        pass.set_bind_group(0, exec.bg0_data.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, exec.bg1_topo.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, exec.bg2_state.as_ref().unwrap(), &[]);
        pass.set_bind_group(3, exec.bg3_aux.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups_indirect(exec.indirect_buffer.as_ref().unwrap(), 12); // Slot 3 (Standard N-length)
    }

    exec.queue.submit(Some(encoder.finish()));
    let _ = exec.device.poll(PollType::Wait);

    // Download standard errors with trimming
    let pad_len = (exec.n - exec.orig_n) / 2;
    let trim_offset = (pad_len as u64) * 4;
    let trim_size = (exec.orig_n as u64) * 4;

    let se_vals = block_on(exec.download_buffer(
        exec.std_errors_buffer.as_ref().unwrap(),
        Some(trim_size),
        Some(trim_offset),
    ))?;

    Some(se_vals.into_iter().map(|v| T::from(v).unwrap()).collect())
}

// Perform a GPU-accelerated LOWESS fit pass.
pub fn fit_pass_gpu<T>(
    x: &[T],
    y: &[T],
    config: &LowessConfig<T>,
) -> Result<IterationResult<T>, LowessError>
where
    T: Float + Debug + Send + Sync + 'static,
{
    #[cfg(feature = "gpu")]
    {
        use pollster::block_on;

        // Global Executor Lock
        let mut guard = GLOBAL_EXECUTOR
            .lock()
            .map_err(|e| LowessError::RuntimeError(format!("GPU mutex poisoned: {}", e)))?;

        if guard.is_none() {
            match block_on(GpuExecutor::new()) {
                Ok(exec) => *guard = Some(exec),
                Err(e) => return Err(LowessError::RuntimeError(format!("GPU init failed: {}", e))),
            }
        }
        let exec = guard.as_mut().unwrap();

        let orig_n = x.len();
        if orig_n > u32::MAX as usize {
            return Err(LowessError::InvalidInput(format!(
                "Dataset too large for GPU backend: {} points (max {})",
                orig_n,
                u32::MAX
            )));
        }

        let delta = config.delta.to_f32().unwrap();
        let window_size =
            (config.fraction.unwrap().to_f32().unwrap() * orig_n as f32).max(1.0) as usize;
        let pad_len = if config.boundary_policy == BoundaryPolicy::NoBoundary {
            0
        } else {
            (window_size / 2).min(orig_n - 1)
        };
        let total_n = orig_n + 2 * pad_len;

        let weight_fn_id = match config.weight_function {
            WeightFunction::Cosine => 0,
            WeightFunction::Epanechnikov => 1,
            WeightFunction::Gaussian => 2,
            WeightFunction::Biweight => 3,
            WeightFunction::Triangle => 4,
            WeightFunction::Tricube => 5,
            WeightFunction::Uniform => 6,
        };

        let fallback_id = config.zero_weight_fallback as u32;

        let boundary_id = match config.boundary_policy {
            BoundaryPolicy::Extend => 0,
            BoundaryPolicy::Reflect => 1,
            BoundaryPolicy::Zero => 2,
            BoundaryPolicy::NoBoundary => 3,
        };

        let gpu_config = GpuConfig {
            n: total_n as u32,
            window_size: window_size as u32,
            weight_function: weight_fn_id,
            zero_weight_fallback: fallback_id,
            fraction: config.fraction.unwrap().to_f32().unwrap(),
            delta,
            median_threshold: 0.0,
            median_center: 0.0,
            is_absolute: 0,
            boundary_policy: boundary_id,
            pad_len: pad_len as u32,
            orig_n: orig_n as u32,
            max_iterations: config.iterations as u32,
            tolerance: config
                .auto_convergence
                .map(|t| t.to_f32().unwrap())
                .unwrap_or(0.0),
        };

        let robustness_id = match config.robustness_method {
            RobustnessMethod::Bisquare => ROBUSTNESS_BISQUARE,
            RobustnessMethod::Huber => ROBUSTNESS_HUBER,
            RobustnessMethod::Talwar => ROBUSTNESS_TALWAR,
        };

        let scaling_id = match config.scaling_method {
            ScalingMethod::MAD => SCALING_MAD,
            ScalingMethod::MAR => SCALING_MAR,
            ScalingMethod::Mean => SCALING_MEAN,
        };

        // Prepare raw data
        let x_f32: Vec<f32> = x.iter().map(|v| v.to_f32().unwrap()).collect();
        let y_f32: Vec<f32> = y.iter().map(|v| v.to_f32().unwrap()).collect();

        exec.reset_buffers(&x_f32, &y_f32, gpu_config, robustness_id, scaling_id);

        // Sort input data (x, y) on GPU
        {
            let mut encoder = exec.device.create_command_encoder(&Default::default());
            exec.record_sort_input(&mut encoder);
            exec.queue.submit(Some(encoder.finish()));
        }

        // Compute anchors and intervals on GPU
        {
            let mut encoder = exec.device.create_command_encoder(&Default::default());
            // 1. Select Anchors (Parallel)
            // 1a. Mark Candidates
            exec.record_pipeline(
                &mut encoder,
                &exec.mark_anchor_candidates_pipeline,
                DispatchMode::Direct(exec.n.div_ceil(256)),
            );
            // 1b. Scan Block
            exec.record_pipeline_with_bg3(
                &mut encoder,
                &exec.scan_block_pipeline,
                DispatchMode::Direct(exec.n.div_ceil(256)),
                exec.bg3_aux.as_ref().unwrap(),
            );
            // 1c. Scan Aux (Sequential)
            exec.record_pipeline_with_bg3(
                &mut encoder,
                &exec.scan_aux_pipeline,
                DispatchMode::Direct(1),
                exec.bg3_aux.as_ref().unwrap(),
            );
            // 1d. Scan Add Base
            exec.record_pipeline_with_bg3(
                &mut encoder,
                &exec.scan_add_base_pipeline,
                DispatchMode::Direct(exec.n.div_ceil(256)),
                exec.bg3_aux.as_ref().unwrap(),
            );

            // 1e. Compact
            exec.record_pipeline_with_bg3(
                &mut encoder,
                &exec.compact_anchors_pipeline,
                DispatchMode::Direct(exec.n.div_ceil(256)),
                exec.bg3_aux.as_ref().unwrap(),
            );
            // 2. Compute intervals (parallel) - Slot 3
            exec.record_pipeline(
                &mut encoder,
                &exec.compute_intervals_pipeline,
                DispatchMode::Direct(exec.n.div_ceil(256)),
            );

            // 3. Initialize Loop State
            exec.record_pipeline(
                &mut encoder,
                &exec.init_loop_state_pipeline,
                DispatchMode::Direct(1),
            );

            // 4. Prepare First Next Pass (Slot 0 for Fit, Slot 3 for Interp/Weights, Slot 6 for Reduction)
            exec.record_pipeline(
                &mut encoder,
                &exec.prepare_next_pass_pipeline,
                DispatchMode::Direct(1),
            );

            exec.queue.submit(Some(encoder.finish()));
        }

        // Initialize weights to 1.0 on GPU
        {
            let mut encoder = exec.device.create_command_encoder(&Default::default());
            exec.record_pipeline(
                &mut encoder,
                &exec.init_weights_pipeline,
                DispatchMode::Direct(exec.n.div_ceil(256)),
            );
            exec.queue.submit(Some(encoder.finish()));
        }

        let mut encoder = exec.device.create_command_encoder(&Default::default());

        for i in 0..=config.iterations {
            // 1. Record Fit (Slot 0 + Slot 3)
            if i > 0 {
                // Copy previous smoothed values to y_prev so convergence check
                // can compare y_smooth (current) vs y_prev (previous).
                let dst_buf = exec.y_prev_buffer.as_ref().unwrap();
                let src_buf = exec.y_smooth_buffer.as_ref().unwrap();
                encoder.copy_buffer_to_buffer(src_buf, 0, dst_buf, 0, src_buf.size());
            }

            exec.record_fit_pass(&mut encoder);

            // 2. Record Convergence Check & Loop Control (Iteration > 0)
            if i > 0 {
                exec.record_convergence_check(&mut encoder); // Reducers use Slot 6
                exec.record_pipeline(
                    &mut encoder,
                    &exec.finalize_convergence_pipeline,
                    DispatchMode::Direct(1),
                );
            }

            // 3. Record Scale estimation if iterations remain
            if i < config.iterations {
                exec.record_robust_scale(&mut encoder, config.scaling_method);
                exec.record_update_weights(&mut encoder); // Slot 3
            }

            // 4. Prepare for next iteration
            exec.record_pipeline(
                &mut encoder,
                &exec.prepare_next_pass_pipeline,
                DispatchMode::Direct(1),
            );
        }

        exec.queue.submit(Some(encoder.finish()));

        let w_config_res = block_on(exec.download_buffer(
            exec.w_config_buffer.as_ref().unwrap(),
            Some(size_of::<WeightConfig>() as u64),
            None,
        ))
        .unwrap();
        // WeightConfig fields (f32/u32 mixed)
        // 0: n, 1: scale, 2: robustness_method, 3: scaling_method, 4: median_center, 5: mean_abs
        println!(
            "Debug: GPU WeightConfig results: n={}, scale={}, mean_abs={}, center={}",
            w_config_res[0], w_config_res[1], w_config_res[5], w_config_res[4]
        );

        let trim_offset = (pad_len as u64) * 4;
        let trim_size = (orig_n as u64) * 4;

        let y_res = block_on(exec.download_buffer(
            exec.y_smooth_buffer.as_ref().unwrap(),
            Some(trim_size),
            Some(trim_offset),
        ))
        .unwrap();

        let w_res = block_on(exec.download_buffer(
            exec.weights_buffer.as_ref().unwrap(),
            Some(trim_size),
            Some(trim_offset),
        ))
        .unwrap();

        // Results are already trimmed by download_buffer
        let y_out: Vec<T> = y_res.into_iter().map(|v| T::from(v).unwrap()).collect();
        let w_out: Vec<T> = w_res.into_iter().map(|v| T::from(v).unwrap()).collect();

        // Compute standard errors/intervals if requested
        let std_errors = if config.return_variance.is_some() {
            compute_intervals_gpu(exec, config)
        } else {
            None
        };

        let iterations_performed = block_on(exec.download_buffer_raw(
            exec.w_config_buffer.as_ref().unwrap(),
            Some(4),
            Some(32),
        )) // offset 32 is 'iteration' in WeightConfig if I updated it correctly
        .map(|raw| cast_slice::<u8, u32>(raw.as_slice())[0])
        .unwrap_or(0) as usize;

        Ok((y_out, std_errors, iterations_performed, w_out))
    }
    #[cfg(not(feature = "gpu"))]
    {
        unimplemented!("GPU feature disabled")
    }
}

// Perform cross-validation to select the best fraction on GPU.
pub fn cross_validate_gpu<T>(
    x: &[T],
    y: &[T],
    fractions: &[T],
    method: CVKind,
    config: &LowessConfig<T>,
) -> (T, Vec<T>)
where
    T: Float + Debug + Send + Sync + 'static,
{
    #[cfg(feature = "gpu")]
    {
        if fractions.is_empty() {
            return (T::zero(), Vec::new());
        }

        let scores: Vec<T> = fractions
            .iter()
            .map(|&frac| {
                let mut cv_buffer = CVBuffer::new();

                // Use the base CV logic for a single fraction
                let (_, s) = method.run(
                    x,
                    y,
                    1, // 1 repetition
                    &[frac],
                    config.cv_seed,
                    |tx, ty, f| {
                        let mut fold_config = config.clone();
                        fold_config.fraction = Some(f);
                        fold_config.cv_fractions = None;

                        let res = fit_pass_gpu(tx, ty, &fold_config);

                        match res {
                            Ok((smoothed, _, _, _)) => smoothed,
                            Err(e) => {
                                panic!("GPU fit failed during CV: {:?}", e);
                            }
                        }
                    },
                    Option::<&mut fn(&[T], &[T], &[T], T) -> Vec<T>>::None,
                    &mut cv_buffer,
                );
                s[0]
            })
            .collect();

        let best_idx = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        (fractions[best_idx], scores)
    }
    #[cfg(not(feature = "gpu"))]
    {
        unimplemented!("GPU feature disabled")
    }
}
