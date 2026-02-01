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
use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::cmp::Ordering::Equal;
use std::fmt::Debug;
use std::mem::size_of;
use std::sync::Mutex;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor,
    Limits, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, Queue,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

// Export dependencies from lowess crate
use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::api::LowessError;
use lowess::internals::engine::executor::{IterationResult, LowessConfig};
use lowess::internals::evaluation::cv::CVKind;
use lowess::internals::evaluation::intervals::IntervalMethod;
use lowess::internals::math::boundary::BoundaryPolicy;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::math::scaling::ScalingMethod;

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
    z_score: f32,
    has_conf: u32,
    has_pred: u32,
    residual_sd: f32,
    n_test: u32,
    seed: u32,
    has_se: u32,
}

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute @workgroup_size(256)
fn init_shuffle(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.n) { return; }
    
    // Key: Random float [0,1)
    // We use seed ^ i to vary the hash. 
    // Usually PCG needs a state update, but stateless hash is fine for shuffling.
    let h = pcg_hash(config.seed ^ i);
    x[i] = f32(h) * 2.3283064365386963e-10; 
    
    // Value: Index (u32 -> f32 bitcast)
    y[i] = bitcast<f32>(i);
}

override PREPARE_MODE: u32 = 0u;
override SORT_MODE: u32 = 0u;
override REDUCE_OP: u32 = 0u;
override REDUCE_SRC: u32 = 0u;
override CONTROL_OP: u32 = 0u;

const CONTROL_RESET_RADIX: u32 = 0u;
const CONTROL_INC_RADIX: u32 = 1u;
const CONTROL_SET_SCALE_MODE: u32 = 2u;
const CONTROL_SET_CENTER_MODE: u32 = 3u;
const CONTROL_FINALIZE_CONV: u32 = 4u;
const CONTROL_PREPARE_NEXT: u32 = 5u;
const CONTROL_INIT_STATE: u32 = 6u;
override FINALIZE_MODE: u32 = 0u;

override CV_TARGET: u32 = 0u;
override WEIGHT_FN: u32 = 99u;

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
    _pad: u32,
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
@group(2) @binding(4) var<storage, read_write> conf_lower: array<f32>;
@group(2) @binding(5) var<storage, read_write> conf_upper: array<f32>;
@group(2) @binding(6) var<storage, read_write> pred_lower: array<f32>;
@group(2) @binding(7) var<storage, read_write> pred_upper: array<f32>;

// Group 3: Aux (Reduction & Weight Config)
@group(3) @binding(0) var<storage, read_write> w_config: WeightConfig;
@group(3) @binding(1) var<storage, read_write> reduction: array<f32>;
@group(3) @binding(2) var<storage, read_write> std_errors: array<f32>;
@group(3) @binding(3) var<storage, read_write> global_histogram: array<atomic<u32>, 256>;
@group(3) @binding(4) var<storage, read_write> global_max_diff: atomic<u32>;
@group(3) @binding(5) var<storage, read_write> scan_block_sums: array<u32>;
@group(3) @binding(6) var<storage, read_write> scan_indices: array<u32>;
@group(3) @binding(7) var<storage, read_write> workgroup_histograms: array<u32>; // [num_workgroups][256]

// Bindings 4-7 in Group 1: CV Global Data
@group(1) @binding(4) var<storage, read_write> x_global: array<f32>;
@group(1) @binding(5) var<storage, read_write> y_global: array<f32>;
@group(1) @binding(6) var<storage, read_write> shuffled_indices: array<u32>;
@group(1) @binding(7) var<storage, read_write> cv_test_mask: array<u32>; // 1 for test, 0 for training

// Workgroup shared memory for scan
var<workgroup> s_scan: array<u32, 256>;

// -----------------------------------------------------------------------------
// Kernel: Loop Control & Dispatch
// -----------------------------------------------------------------------------

// Helper: Calculate kernel weight
fn get_kernel_weight(u: f32, u2: f32) -> f32 {
    var kw = 1.0;
    var fn_id = config.weight_function;
    if (WEIGHT_FN != 99u) {
        fn_id = WEIGHT_FN;
    }
    switch (fn_id) {
        case KERNEL_COSINE: { kw = cos(u * 1.57079632679); }
        case KERNEL_EPANECHNIKOV: { kw = (1.0 - u2); }
        case KERNEL_GAUSSIAN: { kw = exp(-0.5 * u2); }
        case KERNEL_BIWEIGHT: { let v = 1.0 - u2; kw = v * v; }
        case KERNEL_TRIANGLE: { kw = (1.0 - u); }
        case KERNEL_TRICUBE: { let v = 1.0 - u * u2; kw = v * v * v; }
        default: { kw = 1.0; }
    }
    return kw;
}

// Helper: Linear Interpolation
fn get_interpolated_value(x_i: f32, x0: f32, x1: f32, y0: f32, y1: f32) -> f32 {
    let denom = x1 - x0;
    if (abs(denom) < 1e-12) {
        return (y0 + y1) * 0.5;
    } else {
        let t = (x_i - x0) / denom;
        return y0 + t * (y1 - y0);
    }
}

// Helper: Binary Search Window Selection (Tree-Based Neighbor Search)
fn get_adaptive_window(center_idx: u32, x_center: f32) -> vec2<u32> {
    let n = config.n;
    let w_size = config.window_size;

    // Handle degenerate cases
    if (n <= w_size) {
        return vec2(0u, max(0u, n - 1u));
    }
    
    var low = 0u;
    if (center_idx + 1u > w_size) {
        low = center_idx + 1u - w_size;
    }
    
    var high = center_idx;
    let max_left = n - w_size;
    if (high > max_left) {
        high = max_left;
    }

    // Binary search for optimal window start
    while (low < high) {
        let mid = (low + high) / 2u;
        
        let d_left = abs(x_center - x[mid]);
        // Check the point that would be added if we shifted right (x[mid + w_size])
        // Valid because mid < high <= n - w_size, so mid + w_size < n
        let d_right_next = abs(x[mid + w_size] - x_center); 
        
        if (d_left > d_right_next) {
            // Left neighbor is further than the potential right neighbor -> Shift Right
            low = mid + 1u;
        } else {
            high = mid;
        }
    }

    return vec2(low, low + w_size - 1u);
}

@compute @workgroup_size(1)
fn control_pass_generic() {
    switch (CONTROL_OP) {
        case CONTROL_INIT_STATE: {
            w_config.converged = 0u;
            w_config.iteration = 0u;
            w_config.radix_pass = 0u;
        }
        case CONTROL_RESET_RADIX: {
            w_config.radix_pass = 0u;
        }
        case CONTROL_INC_RADIX: {
            w_config.radix_pass = w_config.radix_pass + 1u;
        }
        case CONTROL_SET_SCALE_MODE: {
            w_config.update_mode = MODE_UPDATE_SCALE;
        }
        case CONTROL_SET_CENTER_MODE: {
            w_config.update_mode = MODE_UPDATE_CENTER;
        }
        case CONTROL_FINALIZE_CONV: {
            if (w_config.iteration > 0u) {
                let max_diff_bits = atomicLoad(&global_max_diff);
                let max_diff = bitcast<f32>(max_diff_bits);
                if (max_diff < config.tolerance) {
                    w_config.converged = 1u;
                }
            }
        }
        case CONTROL_PREPARE_NEXT: {
            w_config.iteration = w_config.iteration + 1u;
            atomicStore(&global_max_diff, 0u);
            
            let is_converged = w_config.converged == 1u;
            let anchor_count = w_config.anchor_count;
            
            // Slot 0: fit_anchors (One Workgroup per Anchor)
            if (is_converged) { 
                indirect_args[0] = 0u; 
                indirect_args[1] = 1u;
            } 
            else { 
                if (anchor_count <= 65535u) {
                    indirect_args[0] = anchor_count;
                    indirect_args[1] = 1u;
                } else {
                    indirect_args[0] = 65535u;
                    indirect_args[1] = (anchor_count + 65534u) / 65535u;
                }
            }
            indirect_args[2] = 1u;
            
            indirect_args[2] = 1u;
            
            // Slot 1 (Indices 3,4,5): Compute SE (Workgroup per Point)
            if (is_converged || config.has_se == 0u) { 
                indirect_args[3] = 0u; 
                indirect_args[4] = 1u;
            } 
            else { 
                if (config.n <= 65535u) {
                    indirect_args[3] = config.n;
                    indirect_args[4] = 1u;
                } else {
                    indirect_args[3] = 65535u;
                    indirect_args[4] = (config.n + 65534u) / 65535u;
                }
            }
            indirect_args[4] = 1u;
            indirect_args[5] = 1u;

            // Slot 6: Reduction Kernels (256 threads)
            if (is_converged) { indirect_args[6] = 0u; } 
            else { indirect_args[6] = (config.n + 255u) / 256u; }
            indirect_args[7] = 1u;
            indirect_args[8] = 1u;
        }
        default: {}
    }
}

@compute @workgroup_size(256)
fn clear_histogram(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x < 256u) {
        atomicStore(&global_histogram[global_id.x], 0u);
    }
}

// Bindings 1-3 in Group 1: Test Data (for CV Scoring)
@group(1) @binding(1) var<storage, read_write> x_test: array<f32>;
@group(1) @binding(2) var<storage, read_write> y_test: array<f32>;
@group(1) @binding(3) var<storage, read_write> test_errors: array<f32>;

// Binding 8 in Group 1: CV Results (Global)
// Stores accumulated SSE for each fraction (index 0..N)
@group(1) @binding(8) var<storage, read_write> cv_results: array<f32>;

@compute @workgroup_size(1)
fn accumulate_score() {
    // Current SSE is in reduction[0]
    let sse = reduction[0];
    let idx = w_config.radix_pass;
    
    // Compute RMSE for this fold: sqrt(SSE / n_test)
    let n_test = f32(config.n_test);
    let mse = sse / max(1.0, n_test);
    let rmse = sqrt(mse);
    
    cv_results[idx] += rmse;
}

// -----------------------------------------------------------------------------
// Kernel: Score CV Points
// Performs binary search and interpolation for test points against fitted training data
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn score_cv_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x; // Index into test set
    // test_errors length is number of test points
    let n_test = arrayLength(&test_errors);
    if (j >= n_test) { return; }

    let xt = x_test[j];
    let yt = y_test[j];
    
    // x and y_smooth are the TRAINING data (sorted)
    let n_train = config.n;
    var pred = 0.0;
    let start_idx = config.pad_len;
    let end_idx = config.pad_len + config.orig_n;
    
    // Bounds check / Extrapolation (using original unpadded range)
    if (config.orig_n == 0u) {
        pred = 0.0;
    } else if (n_train > 0u && xt <= x[start_idx]) {
        pred = y_smooth[start_idx];
    } else if (n_train > 0u && xt >= x[end_idx - 1u]) {
        pred = y_smooth[end_idx - 1u];
    } else {
        // Binary search for bracket [L, R] within original range
        var left = start_idx;
        var right = end_idx - 1u;
        
        // Loop limit for safety
        for (var k = 0u; k < 64u; k++) {
            if (right - left <= 1u) { break; }
            let mid = (left + right) / 2u;
            if (x[mid] <= xt) {
                left = mid;
            } else {
                right = mid;
            }
        }
        
        pred = get_interpolated_value(xt, x[left], x[right], y_smooth[left], y_smooth[right]);
    }
    
    let err = yt - pred;
    test_errors[j] = err * err;
}

// -----------------------------------------------------------------------------
// Kernel: Sum SSE Reduction
// Reduces test_errors to partial sums in reduction buffer
// -----------------------------------------------------------------------------
var<workgroup> s_reduce: array<f32, 256>;


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

struct FitSums {
    w: f32,
    wx: f32,
    wxx: f32,
    wy: f32,
    wxy: f32,
    y_window: f32,
}
var<workgroup> s_fit: array<FitSums, 256>;

@compute @workgroup_size(256)
fn fit_anchors(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // 2D dispatch support: (N % 65535, N / 65535, 1) to support >65535 anchors
    let anchor_id = workgroup_id.x + workgroup_id.y * 65535u;
    let tid = local_id.x;
    
    let num_anchors_explicit = w_config.anchor_count;

    if (anchor_id >= num_anchors_explicit) {
        return;
    }

    let i = anchor_indices[anchor_id];
    let x_i = x[i];
    
    let win = get_adaptive_window(i, x_i);
    let left = win.x;
    let right = win.y;

    // Check degenerate window
    let d_max = max(abs(x_i - x[left]), abs(x_i - x[right]));
    if (d_max <= 1e-12) {
        if (tid == 0u) {
            anchor_output[anchor_id] = y[i];
        }
        return;
    }

    // Initialize local sums
    var my_w = 0.0;
    var my_wx = 0.0;
    var my_wxx = 0.0;
    var my_wy = 0.0;
    var my_wxy = 0.0;
    var my_y_window = 0.0;

    let d_max_val = max(d_max, 1e-12);
    let inv_d_max = 1.0 / d_max_val;
    let h1 = 0.001 * d_max_val;
    let h9 = 0.999 * d_max_val;
    
    let iter = w_config.iteration;

    // Parallel accumulation loop
    // Stride is 256 (workgroup size)
    for (var k = left + tid; k <= right; k += 256u) {
        let xj = x[k];
        let yj = y[k];
        let rel_x = xj - x_i;
        let dist = abs(rel_x);
        
        my_y_window += yj;
        
        if (dist <= h9) {
            var kernel_w = 1.0;
            if (dist > h1) {
                let u = dist * inv_d_max;
                let u2 = u * u;
                kernel_w = get_kernel_weight(u, u2);
            }
            
            var rw = 1.0;
            if (iter > 0u) {
                rw = robustness_weights[k];
            }
            
            let combined_w = rw * kernel_w;
            my_w += combined_w;
            my_wx += combined_w * rel_x;
            my_wxx += combined_w * rel_x * rel_x;
            my_wy += combined_w * yj;
            my_wxy += combined_w * rel_x * yj;
        }
    }

    // Load to shared memory
    s_fit[tid].w = my_w;
    s_fit[tid].wx = my_wx;
    s_fit[tid].wxx = my_wxx;
    s_fit[tid].wy = my_wy;
    s_fit[tid].wxy = my_wxy;
    s_fit[tid].y_window = my_y_window;
    
    workgroupBarrier();

    // Parallel Reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_fit[tid].w += s_fit[tid + s].w;
            s_fit[tid].wx += s_fit[tid + s].wx;
            s_fit[tid].wxx += s_fit[tid + s].wxx;
            s_fit[tid].wy += s_fit[tid + s].wy;
            s_fit[tid].wxy += s_fit[tid + s].wxy;
            s_fit[tid].y_window += s_fit[tid + s].y_window;
        }
        workgroupBarrier();
    }

    // Final Solve (Thread 0)
    if (tid == 0u) {
        let sum_w = s_fit[0].w;
        let sum_wx = s_fit[0].wx;
        let sum_wxx = s_fit[0].wxx;
        let sum_wy = s_fit[0].wy;
        let sum_wxy = s_fit[0].wxy;
        let sum_y_window = s_fit[0].y_window;
        
        let TOL: f32 = 1e-12;
        if (sum_w < TOL) {
             switch (config.zero_weight_fallback) {
                case FALLBACK_RETURN_ORIGINAL: { anchor_output[anchor_id] = y[i]; }
                default: { 
                    let w_size = f32(right - left + 1u);
                    anchor_output[anchor_id] = sum_y_window / max(1.0, w_size); 
                }
            }
        } else {
             let x_mean = sum_wx / sum_w;
             let y_mean = sum_wy / sum_w;
             let variance = sum_wxx - (sum_wx * sum_wx) / sum_w;
             
             let abs_tol: f32 = 1e-7;
             let rel_tol: f32 = 1.1920929e-7 * d_max_val * d_max_val;
             let tol = max(abs_tol, rel_tol);

             if (variance <= tol) {
                 anchor_output[anchor_id] = y_mean;
             } else {
                 let covariance = sum_wxy - (sum_wx * sum_wy) / sum_w;
                 let slope = covariance / variance;
                 let intercept = y_mean - slope * x_mean;
                 let fitted = intercept;
                 
                  if (fitted == fitted && abs(fitted) < 1e15) {
                    anchor_output[anchor_id] = fitted;
                } else {
                    anchor_output[anchor_id] = y_mean;
                }
             }
        }
        
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
@compute @workgroup_size(256)
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
        fitted = get_interpolated_value(x_i, x_l, x_r, y_l, y_r);
    }

    y_smooth[i] = fitted;
    residuals[i] = y[i] - fitted;
}

// -----------------------------------------------------------------------------
// Kernel 3: Update Weights
// Dispatched with N threads
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
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
fn reduce_generic(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let i = global_id.x;
    var val = 0.0;
    
    var n_eff = config.n;
    if (REDUCE_SRC == 1u) { n_eff = config.orig_n; }
    if (REDUCE_SRC == 2u) { n_eff = config.n_test; }

    if (i < n_eff) {
        switch (REDUCE_SRC) {
            case 0u: { val = abs(residuals[i]); }
            case 1u: {
                let pad = config.pad_len;
                let r = residuals[i + pad];
                val = r * r;
            }
            case 2u: { val = test_errors[i]; }
            case 3u: { val = abs(y_smooth[i] - y_prev[i]); }
            default: { }
        }
    }
    
    scratch[lid.x] = val;
    workgroupBarrier();
    
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            if (REDUCE_OP == 0u) {
                scratch[lid.x] += scratch[lid.x + s];
            } else {
                scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + s]);
            }
        }
        workgroupBarrier();
    }
    
    if (lid.x == 0u) {
        reduction[wid.x] = scratch[0];
    }
}

@compute @workgroup_size(1)
fn finalize_reduction_generic() {
    let n = config.n;
    var sum = 0.0;
    let num_workgroups = (n + 255u) / 256u;
    for (var i: u32 = 0u; i < num_workgroups; i = i + 1u) {
        sum = sum + reduction[i];
    }
    let val = sum / f32(max(1u, n));
    
    if (FINALIZE_MODE == 0u) {
        // Mode 0: Scale
        w_config.scale = max(val, 1e-12);
    } else {
        // Mode 1: Mean Abs
        w_config.mean_abs = val;
    }
    reduction[0] = sum;
}

// -----------------------------------------------------------------------------
// Kernel 5: Standard Errors
// Dispatched with N threads
// -----------------------------------------------------------------------------
struct SeSums {
    w: f32,
    wr2: f32,
}
var<workgroup> s_se: array<SeSums, 256>;

@compute @workgroup_size(256)
fn compute_se(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // 2D Dispatch: Workgroup per Point
    let i = workgroup_id.x + workgroup_id.y * 65535u;
    let tid = local_id.x;
    let n = config.n;
    
    if (i >= n) { return; }

    let x_i = x[i];
    
    let win = get_adaptive_window(i, x_i);
    let left = win.x;
    let right = win.y;

    let d_max = max(abs(x_i - x[left]), abs(x_i - x[right]));
    if (d_max <= 1e-12) {
        if (tid == 0u) {
            std_errors[i] = 0.0;
        }
        return;
    }

    var my_w = 0.0;
    var my_wr2 = 0.0;
    let d_max_val = max(d_max, 1e-9);
    let inv_d_max = 1.0 / d_max_val;
    
    for (var k = left + tid; k <= right; k += 256u) {
        let xj = x[k];
        let smoothed_j = y_smooth[k];
        let yj = y[k];
        let rw = robustness_weights[k];
        
        let r = yj - smoothed_j;
        let dist = abs(xj - x_i);
        let u = dist * inv_d_max;
        
        if (u < 1.0) {
            let u2 = u * u;
            let kernel_w = get_kernel_weight(u, u2);
            
            let combined_w = rw * kernel_w;
            my_w += combined_w;
            my_wr2 += combined_w * r * r;
        }
    }

    s_se[tid].w = my_w;
    s_se[tid].wr2 = my_wr2;
    workgroupBarrier();
    
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_se[tid].w += s_se[tid + s].w;
            s_se[tid].wr2 += s_se[tid + s].wr2;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let sum_w = s_se[0].w;
        let sum_wr2 = s_se[0].wr2;
        
        let LINEAR_PARAMS = 2.0;
        if (sum_w > LINEAR_PARAMS + 1e-6) {
            let df = sum_w - LINEAR_PARAMS;
            let variance = sum_wr2 / df;
            
            // Find kernel weight for current point (distance = 0)
            // u=0 -> kernel_weight=1.0 usually but let's call function
            let w_idx_kern = get_kernel_weight(0.0, 0.0);
            let w_idx = w_idx_kern * robustness_weights[i];
            
            let leverage = w_idx / sum_w;
            std_errors[i] = sqrt(variance * leverage);
        } else {
            std_errors[i] = 0.0;
        }
    }
}

@compute @workgroup_size(256)
fn compute_interval_bounds(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.n) { return; }

    let ys = y_smooth[i];
    let se = std_errors[i];
    let rsd = config.residual_sd;
    let z = config.z_score;

    if (config.has_conf != 0u) {
        conf_lower[i] = ys - z * se;
        conf_upper[i] = ys + z * se;
    }

    if (config.has_pred != 0u) {
        let pred_se = sqrt(se * se + rsd * rsd);
        pred_lower[i] = ys - z * pred_se;
        pred_upper[i] = ys + z * pred_se;
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

@compute @workgroup_size(1)
fn compute_residual_sd() {
    let sse = reduction[0];
    let n = f32(config.orig_n);
    var sd = 0.0;
    if (n > 2.0) {
        sd = sqrt(sse / (n - 2.0));
    }
    // Store in reduction[1] as temp
    reduction[1] = sd;
}

@compute @workgroup_size(1)
fn update_dispatch() {
    let n = config.n;
    let converged = w_config.converged;
    
    var d_n = 0u;
    var d_anchors = 0u;

    if (converged == 0u) {
        d_n = (n + 255u) / 256u;
        // d_anchors logic moved to below for 2D dispatch
    }
    
    // Slot 0: Fit Anchors (Workgroup per Anchor)
    let limit = 65535u;
    if (w_config.anchor_count <= limit) {
        indirect_args[0] = w_config.anchor_count;
        indirect_args[1] = 1u;
    } else {
        indirect_args[0] = limit;
        indirect_args[1] = (w_config.anchor_count + limit - 1u) / limit;
    }
    indirect_args[2] = 1u;

    // Slot 2: Interpolate
    indirect_args[6] = d_n;
    indirect_args[7] = 1u;
    indirect_args[8] = 1u;
    
    // Slot 5: Update Weights
    indirect_args[15] = d_n;
    indirect_args[16] = 1u;
    indirect_args[17] = 1u;
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

@compute @workgroup_size(256)
fn prepare_reduction_generic(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.n) { return; }

    var val: f32 = 0.0;
    switch (PREPARE_MODE) {
        case 0u: { val = abs(residuals[i]); } // MAR
        case 1u: { val = residuals[i]; }     // Signed (for center)
        case 2u: { val = abs(residuals[i] - w_config.median_center); } // MAD
        default: { }
    }
    reduction[i] = val;
}

// Radix histogram: count occurrences of each 8-bit digit per workgroup
@compute @workgroup_size(256)
fn sort_histogram_block(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>,
                        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let is_x = SORT_MODE == 1u;
    let n = select(config.n, config.orig_n, is_x);
    let pad = select(0u, config.pad_len, is_x);
    let radix_pass = w_config.radix_pass;
    let shift = radix_pass * 8u;
    let wid = workgroup_id.x;
    let lid = local_id.x;

    // Initialize local histogram
    atomicStore(&local_histogram[lid], 0u);
    workgroupBarrier();
    
    // Count digits in our tile
    let i = global_id.x;
    if (i < n) {
        var val: u32;
        if (is_x) {
            val = f32_to_sortable_u32(x[i + pad]);
        } else {
            val = f32_to_sortable_u32(reduction[i]);
        }
        let digit = (val >> shift) & 0xFFu;
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();
    
    // Write workgroup histogram to global memory
    // Each thread writes one bin
    if (lid < 256u) {
        let count = atomicLoad(&local_histogram[lid]);
        workgroup_histograms[wid * 256u + lid] = count;
        
        // Also accumulate into global histogram (for the absolute offsets)
        if (count > 0u) {
            atomicAdd(&global_histogram[lid], count);
        }
    }
}

// Prefix sum on workgroup histograms (single workgroup or multi-pass)
// For simplicity and since num_workgroups is ~400 for 100k, 
// we scan the "columns" of workgroup_histograms.
@compute @workgroup_size(256)
fn radix_prefix_sum(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let digit = local_id.x; // One thread per digit
    let num_workgroups = (select(config.n, config.orig_n, SORT_MODE == 1u) + 255u) / 256u;
    
    // Global base offset for this digit (sum of all counts for digits < digit)
    var global_base = 0u;
    for (var d = 0u; d < digit; d = d + 1u) {
        global_base += atomicLoad(&global_histogram[d]);
    }
    
    // Inclusive scan over workgroup counts for this digit
    var running_sum = global_base;
    for (var w = 0u; w < num_workgroups; w = w + 1u) {
        let idx = w * 256u + digit;
        let count = workgroup_histograms[idx];
        workgroup_histograms[idx] = running_sum; // Store exclusive prefix sum
        running_sum += count;
    }
}

// Shared memory for digits and local offsets (one per thread)
var<workgroup> s_digits: array<u32, 256>;
var<workgroup> s_offsets: array<u32, 256>;
var<workgroup> s_hist_local: array<u32, 256>;

// Parallel Scatter using pre-calculated offsets
@compute @workgroup_size(256)
fn sort_scatter_parallel(@builtin(global_invocation_id) global_id: vec3<u32>,
                         @builtin(local_invocation_id) local_id: vec3<u32>,
                         @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let is_x = SORT_MODE == 1u;
    let n = select(config.n, config.orig_n, is_x);
    let pad = select(0u, config.pad_len, is_x);
    let radix_pass = w_config.radix_pass;
    let shift = radix_pass * 8u;
    let wid = workgroup_id.x;
    let lid = local_id.x;

    let i = global_id.x;
    var has_val = false;
    var my_digit = 0u;
    var my_val: u32 = 0u;

    if (i < n) {
        if (is_x) {
            my_val = f32_to_sortable_u32(x[i + pad]);
        } else {
            my_val = f32_to_sortable_u32(reduction[i]);
        }
        my_digit = (my_val >> shift) & 0xFFu;
        has_val = true;
    }

    // Store digits for sequential pass
    s_digits[lid] = select(256u, my_digit, has_val);
    workgroupBarrier();

    // Thread 0: Sequential stable scan within workgroup
    if (lid == 0u) {
        // Clear local counts
        for (var d = 0u; d < 256u; d++) {
            s_hist_local[d] = 0u;
        }
        // Scan elements
        for (var j = 0u; j < 256u; j++) {
            let digit = s_digits[j];
            if (digit < 256u) {
                s_offsets[j] = s_hist_local[digit];
                s_hist_local[digit]++;
            }
        }
    }
    workgroupBarrier();

    if (has_val) {
        // Global position = global_workgroup_base[wid][digit] + local_offset[digit]
        let local_offset = s_offsets[lid];
        let global_workgroup_base = workgroup_histograms[wid * 256u + my_digit];
        let pos = global_workgroup_base + local_offset;

        if (is_x) {
            let val_x = x[i + pad];
            let val_y = y[i + pad];
            y_smooth[pos + pad] = val_x;
            residuals[pos + pad] = val_y;
        } else {
            let element = reduction[i];
            reduction[524288u + pos] = element;
        }
    }
}

// Copy back from upper half to lower half
@compute @workgroup_size(256)
fn sort_copy_back_generic(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let is_x = SORT_MODE == 1u;
    
    if (is_x) {
        if (i < config.orig_n) {
            let pad = config.pad_len;
            let idx = i + pad;
            x[idx] = y_smooth[idx];
            y[idx] = residuals[idx];
        }
    } else {
        if (i < config.n) {
            reduction[i] = reduction[524288u + i];
        }
    }
}

// Select median from sorted reduction
// Result stored in reduction[1048575]
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
// CV Partitioning Kernels
// -----------------------------------------------------------------------------

@compute @workgroup_size(256)
fn cv_prepare_mask(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.orig_n + config.n_test) { return; }
    cv_test_mask[i] = 0u;
}

@compute @workgroup_size(256)
fn cv_mark_test_indices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;
    if (j >= config.n_test) { return; }
    
    // config.z_score reused as test_start for CV
    let test_start = bitcast<u32>(config.z_score);
    let idx = shuffled_indices[test_start + j];
    cv_test_mask[idx] = 1u;
}

// Reuses scan_indices as flags for compaction
@compute @workgroup_size(256)
fn cv_prepare_compact_flags(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.orig_n + config.n_test) { return; }
    // Flag is 1 if mask matches target
    scan_indices[i] = select(0u, 1u, cv_test_mask[i] == CV_TARGET);
}

@compute @workgroup_size(256)
fn cv_compact_data(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.orig_n + config.n_test) { return; }
    
    // Only compact if mask matches target
    if (cv_test_mask[i] == CV_TARGET) {
        let write_idx = scan_indices[i] - 1u;
        // If Target is Train (0), apply padded offset.
        // If Target is Test (1), no offset.
        let offset = select(config.pad_len, 0u, CV_TARGET == 1u);
        
        x[write_idx + offset] = x_global[i];
        y[write_idx + offset] = y_global[i];
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
    pub z_score: f32,
    pub has_conf: u32,
    pub has_pred: u32,
    pub residual_sd: f32,
    pub n_test: u32,
    pub seed: u32,
    pub has_se: u32,
    // Start Padding (Ensure 16-byte alignment for Uniform Buffer)
    // Current size: 84 bytes. Need 96 bytes. (12 bytes padding)
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
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
    pub _pad: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum ControlOp {
    ResetRadix = 0,
    IncRadix = 1,
    SetScaleMode = 2,
    SetCenterMode = 3,
    FinalizeConv = 4,
    PrepareNext = 5,
    InitState = 6,
}

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum PrepareOp {
    #[allow(clippy::upper_case_acronyms)]
    MAR = 0,
    Signed = 1,
    #[allow(clippy::upper_case_acronyms)]
    MAD = 2,
}

pub struct GpuPipelines {
    pub fit_pipelines: [ComputePipeline; 7],
    pub interpolate_pipeline: ComputePipeline,
    pub weight_pipeline: ComputePipeline,
    pub reduce_max_diff_pipeline: ComputePipeline,
    pub reduce_sum_abs_pipeline: ComputePipeline,
    pub finalize_scale_pipeline: ComputePipeline,
    pub init_weights_pipeline: ComputePipeline,
    pub compute_intervals_pipeline: ComputePipeline,
    pub prepare_pipelines: [ComputePipeline; 3],
    pub radix_histogram_pipeline: ComputePipeline,
    pub radix_scan_histograms_pipeline: ComputePipeline,
    pub radix_scatter_pipeline: ComputePipeline,
    pub radix_copy_back_pipeline: ComputePipeline,
    pub select_median_pipeline: ComputePipeline,
    pub sort_x_histogram_pipeline: ComputePipeline,
    pub sort_x_scan_histograms_pipeline: ComputePipeline,
    pub sort_x_scatter_pipeline: ComputePipeline,
    pub sort_x_copy_back_pipeline: ComputePipeline,
    pub se_pipeline: ComputePipeline,
    pub pad_pipeline: ComputePipeline,
    pub update_scale_config_pipeline: ComputePipeline,
    pub clear_histogram_pipeline: ComputePipeline,
    pub control_pipelines: [ComputePipeline; 7],
    pub scan_block_pipeline: ComputePipeline,
    pub scan_add_base_pipeline: ComputePipeline,
    pub scan_aux_pipeline: ComputePipeline,
    pub compact_anchors_pipeline: ComputePipeline,
    pub mark_anchor_candidates_pipeline: ComputePipeline,
    pub finalize_sum_pipeline: ComputePipeline,
    pub score_cv_points_pipeline: ComputePipeline,
    pub sum_sse_reduction_pipeline: ComputePipeline,
    pub interval_bounds_pipeline: ComputePipeline,
    pub sum_residuals_squared_pipeline: ComputePipeline,
    pub cv_prepare_mask_pipeline: ComputePipeline,
    pub cv_mark_test_indices_pipeline: ComputePipeline,
    pub cv_prepare_compact_flags_train_pipeline: ComputePipeline,
    pub cv_prepare_compact_flags_test_pipeline: ComputePipeline,
    pub cv_compact_training_pipeline: ComputePipeline,
    pub cv_compact_test_pipeline: ComputePipeline,
    pub accumulate_score_pipeline: ComputePipeline,
    pub init_shuffle_pipeline: ComputePipeline,
    pub compute_residual_sd_pipeline: ComputePipeline,
    pub update_dispatch_pipeline: ComputePipeline,
}

pub struct GpuBuffers {
    // Group 0
    pub config_buffer: Option<Buffer>,
    pub x_buffer: Option<Buffer>,
    pub y_buffer: Option<Buffer>,
    pub anchor_indices_buffer: Option<Buffer>,
    pub anchor_output_buffer: Option<Buffer>,
    pub indirect_buffer: Option<Buffer>,

    // Group 1
    pub interval_map_buffer: Option<Buffer>,

    // Group 2
    pub weights_buffer: Option<Buffer>,
    pub y_smooth_buffer: Option<Buffer>,
    pub y_prev_buffer: Option<Buffer>,
    pub residuals_buffer: Option<Buffer>,
    pub histogram_buffer: Option<Buffer>,
    pub conf_lower_buffer: Option<Buffer>,
    pub conf_upper_buffer: Option<Buffer>,
    pub pred_lower_buffer: Option<Buffer>,
    pub pred_upper_buffer: Option<Buffer>,

    // Group 3
    pub w_config_buffer: Option<Buffer>,
    pub reduction_buffer: Option<Buffer>,
    pub std_errors_buffer: Option<Buffer>,
    pub global_max_diff_buffer: Option<Buffer>,
    pub median_buffer: Option<Buffer>,
    pub scan_block_sums_buffer: Option<Buffer>,
    pub scan_indices_buffer: Option<Buffer>,
    pub workgroup_histograms_buffer: Option<Buffer>,

    // Group 4
    pub x_test_buffer: Option<Buffer>,
    pub y_test_buffer: Option<Buffer>,
    pub test_errors_buffer: Option<Buffer>,

    // Group 5
    pub x_global_buffer: Option<Buffer>,
    pub y_global_buffer: Option<Buffer>,
    pub shuffled_indices_buffer: Option<Buffer>,
    pub cv_test_mask_buffer: Option<Buffer>,
    pub cv_results_buffer: Option<Buffer>,

    // Staging
    pub staging_buffer: Option<Buffer>,
}

impl GpuBuffers {
    pub fn ensure_capacity(
        device: &Device,
        label: &str,
        buffer_opt: &mut Option<Buffer>,
        size_required: u64,
        usage: BufferUsages,
    ) -> bool {
        let mut created_new = false;
        if buffer_opt
            .as_ref()
            .is_some_and(|buf| buf.size() < size_required)
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
}

pub struct GpuExecutor {
    pub device: Device,
    pub queue: Queue,

    // Pipelines
    pub pipelines: GpuPipelines,
    pub buffers: GpuBuffers,

    // Bind Groups
    bg0_data: Option<BindGroup>,
    bg1_topo: Option<BindGroup>,
    bg2_state: Option<BindGroup>,
    bg3_aux: Option<BindGroup>,
    bg3_median: Option<BindGroup>, // Alternate BG3 with median_buffer as target
    bg0_test: Option<BindGroup>,   // BG0 variants for CV (Test Data as Target)

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

        let limits = Limits {
            max_storage_buffers_per_shader_stage: 64,
            ..Default::default()
        };

        let (device, queue): (Device, Queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("LOWESS Device"),
                required_features: Features::empty(),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Device error: {:?}", e))?;

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("LOWESS Shader"),
            source: ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // Layout Helpers
        let layout_entry = |binding: u32, read_only: bool| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let uniform_entry = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // Layouts
        let bind_group_layout_0 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BG0 Data"),
            entries: &[
                uniform_entry(0),
                layout_entry(1, false),
                layout_entry(2, false),
                layout_entry(3, false),
                layout_entry(4, false),
                layout_entry(5, false),
            ],
        });

        let bind_group_layout_1 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BG1 Topo"),
            entries: &[
                layout_entry(0, false),
                layout_entry(1, false),
                layout_entry(2, false),
                layout_entry(3, false),
                layout_entry(4, false),
                layout_entry(5, false),
                layout_entry(6, false),
                layout_entry(7, false),
                layout_entry(8, false),
            ],
        });

        let bind_group_layout_2 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BG2 State"),
            entries: &[
                layout_entry(0, false),
                layout_entry(1, false),
                layout_entry(2, false),
                layout_entry(3, false),
                layout_entry(4, false),
                layout_entry(5, false),
                layout_entry(6, false),
                layout_entry(7, false),
            ],
        });

        let bind_group_layout_3 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BG3 Aux"),
            entries: &[
                layout_entry(0, false),
                layout_entry(1, false),
                layout_entry(2, false),
                layout_entry(3, false),
                layout_entry(4, false),
                layout_entry(5, false),
                layout_entry(6, false),
                layout_entry(7, false),
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

        let cp = |label: &str, entry: &str, constants: &[(&str, f64)]| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        let cps = |entry: &str| cp(entry, entry, &[]);

        Ok(Self {
            pipelines: GpuPipelines {
                fit_pipelines: [0, 1, 2, 3, 4, 5, 6]
                    .map(|i| cp("fit_anchors", "fit_anchors", &[("WEIGHT_FN", i as f64)])),
                interpolate_pipeline: cps("interpolate"),
                weight_pipeline: cps("update_weights"),
                reduce_max_diff_pipeline: cp(
                    "reduce_max_diff",
                    "reduce_generic",
                    &[("REDUCE_OP", 1.0), ("REDUCE_SRC", 3.0)],
                ),
                reduce_sum_abs_pipeline: cp(
                    "reduce_sum_abs",
                    "reduce_generic",
                    &[("REDUCE_OP", 0.0), ("REDUCE_SRC", 0.0)],
                ),
                finalize_scale_pipeline: cp(
                    "finalize_scale",
                    "finalize_reduction_generic",
                    &[("FINALIZE_MODE", 0.0)],
                ),
                finalize_sum_pipeline: cp(
                    "finalize_sum",
                    "finalize_reduction_generic",
                    &[("FINALIZE_MODE", 1.0)],
                ),
                score_cv_points_pipeline: cps("score_cv_points"),
                sum_sse_reduction_pipeline: cp(
                    "sum_sse_reduction",
                    "reduce_generic",
                    &[("REDUCE_OP", 0.0), ("REDUCE_SRC", 2.0)],
                ),
                sum_residuals_squared_pipeline: cp(
                    "sum_residuals_squared",
                    "reduce_generic",
                    &[("REDUCE_OP", 0.0), ("REDUCE_SRC", 1.0)],
                ),
                cv_prepare_mask_pipeline: cps("cv_prepare_mask"),
                cv_mark_test_indices_pipeline: cps("cv_mark_test_indices"),
                cv_prepare_compact_flags_train_pipeline: cp(
                    "cv_prepare_compact_flags_train",
                    "cv_prepare_compact_flags",
                    &[("CV_TARGET", 0.0)],
                ),
                cv_prepare_compact_flags_test_pipeline: cp(
                    "cv_prepare_compact_flags_test",
                    "cv_prepare_compact_flags",
                    &[("CV_TARGET", 1.0)],
                ),
                cv_compact_training_pipeline: cp(
                    "cv_compact_training",
                    "cv_compact_data",
                    &[("CV_TARGET", 0.0)],
                ),
                cv_compact_test_pipeline: cp(
                    "cv_compact_test",
                    "cv_compact_data",
                    &[("CV_TARGET", 1.0)],
                ),
                accumulate_score_pipeline: cps("accumulate_score"),
                init_shuffle_pipeline: cps("init_shuffle"),
                compute_residual_sd_pipeline: cps("compute_residual_sd"),
                update_dispatch_pipeline: cps("update_dispatch"),
                interval_bounds_pipeline: cps("compute_interval_bounds"),
                init_weights_pipeline: cps("init_weights"),
                compute_intervals_pipeline: cps("compute_intervals"),
                prepare_pipelines: [
                    cp(
                        "prepare_mar_residuals",
                        "prepare_reduction_generic",
                        &[("PREPARE_MODE", 0.0)],
                    ),
                    cp(
                        "prepare_residuals_signed",
                        "prepare_reduction_generic",
                        &[("PREPARE_MODE", 1.0)],
                    ),
                    cp(
                        "prepare_mad_residuals",
                        "prepare_reduction_generic",
                        &[("PREPARE_MODE", 2.0)],
                    ),
                ],
                radix_histogram_pipeline: cp(
                    "radix_histogram",
                    "sort_histogram_block",
                    &[("SORT_MODE", 0.0)],
                ),
                radix_scan_histograms_pipeline: cps("radix_prefix_sum"),
                radix_scatter_pipeline: cp(
                    "radix_scatter",
                    "sort_scatter_parallel",
                    &[("SORT_MODE", 0.0)],
                ),
                radix_copy_back_pipeline: cp(
                    "radix_copy_back",
                    "sort_copy_back_generic",
                    &[("SORT_MODE", 0.0)],
                ),
                select_median_pipeline: cps("select_median"),
                sort_x_histogram_pipeline: cp(
                    "sort_x_histogram",
                    "sort_histogram_block",
                    &[("SORT_MODE", 1.0)],
                ),
                sort_x_scan_histograms_pipeline: cp(
                    "sort_x_scan_histograms",
                    "radix_prefix_sum",
                    &[("SORT_MODE", 1.0)],
                ),
                sort_x_scatter_pipeline: cp(
                    "sort_x_scatter",
                    "sort_scatter_parallel",
                    &[("SORT_MODE", 1.0)],
                ),
                sort_x_copy_back_pipeline: cp(
                    "sort_x_copy_back",
                    "sort_copy_back_generic",
                    &[("SORT_MODE", 1.0)],
                ),
                se_pipeline: cps("compute_se"),
                pad_pipeline: cps("pad_data"),
                update_scale_config_pipeline: cps("update_scale_config"),
                clear_histogram_pipeline: cps("clear_histogram"),
                control_pipelines: [
                    cp(
                        "control_reset_radix",
                        "control_pass_generic",
                        &[("CONTROL_OP", 0.0)],
                    ),
                    cp(
                        "control_inc_radix",
                        "control_pass_generic",
                        &[("CONTROL_OP", 1.0)],
                    ),
                    cp(
                        "control_set_scale_mode",
                        "control_pass_generic",
                        &[("CONTROL_OP", 2.0)],
                    ),
                    cp(
                        "control_set_center_mode",
                        "control_pass_generic",
                        &[("CONTROL_OP", 3.0)],
                    ),
                    cp(
                        "control_finalize_conv",
                        "control_pass_generic",
                        &[("CONTROL_OP", 4.0)],
                    ),
                    cp(
                        "control_prepare_next",
                        "control_pass_generic",
                        &[("CONTROL_OP", 5.0)],
                    ),
                    cp(
                        "control_init",
                        "control_pass_generic",
                        &[("CONTROL_OP", 6.0)],
                    ),
                ],
                scan_block_pipeline: cps("scan_block"),
                scan_aux_pipeline: cps("scan_aux_serial"),
                scan_add_base_pipeline: cps("scan_add_base"),
                compact_anchors_pipeline: cps("compact_anchors"),
                mark_anchor_candidates_pipeline: cps("mark_anchor_candidates"),
            },
            device,
            queue,
            buffers: GpuBuffers {
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
                conf_lower_buffer: None,
                conf_upper_buffer: None,
                pred_lower_buffer: None,
                pred_upper_buffer: None,
                w_config_buffer: None,
                reduction_buffer: None,
                std_errors_buffer: None,
                global_max_diff_buffer: None,
                median_buffer: None,
                scan_block_sums_buffer: None,
                scan_indices_buffer: None,
                workgroup_histograms_buffer: None,
                x_test_buffer: None,
                y_test_buffer: None,
                test_errors_buffer: None,
                x_global_buffer: None,
                y_global_buffer: None,
                shuffled_indices_buffer: None,
                cv_test_mask_buffer: None,
                cv_results_buffer: None,
                staging_buffer: None,
            },
            bg0_data: None,
            bg1_topo: None,
            bg2_state: None,
            bg3_aux: None,
            bg3_median: None,
            bg0_test: None,
            n: 0,
            orig_n: 0,
            num_anchors: 0,
        })
    }

    fn build_bg(&self, label: &str, layout_idx: u32, buffers: &[&Buffer]) -> BindGroup {
        let entries: Vec<BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(label),
            layout: &self.pipelines.fit_pipelines[0].get_bind_group_layout(layout_idx),
            entries: &entries,
        })
    }

    fn create_bg0(&self, x: &Buffer, y: &Buffer) -> BindGroup {
        self.build_bg(
            "BG0",
            0,
            &[
                self.buffers.config_buffer.as_ref().unwrap(),
                x,
                y,
                self.buffers.anchor_indices_buffer.as_ref().unwrap(),
                self.buffers.anchor_output_buffer.as_ref().unwrap(),
                self.buffers.indirect_buffer.as_ref().unwrap(),
            ],
        )
    }

    fn create_bg1(&self) -> BindGroup {
        self.build_bg(
            "BG1",
            1,
            &[
                self.buffers.interval_map_buffer.as_ref().unwrap(),
                self.buffers.x_test_buffer.as_ref().unwrap(),
                self.buffers.y_test_buffer.as_ref().unwrap(),
                self.buffers.test_errors_buffer.as_ref().unwrap(),
                self.buffers.x_global_buffer.as_ref().unwrap(),
                self.buffers.y_global_buffer.as_ref().unwrap(),
                self.buffers.shuffled_indices_buffer.as_ref().unwrap(),
                self.buffers.cv_test_mask_buffer.as_ref().unwrap(),
                self.buffers.cv_results_buffer.as_ref().unwrap(),
            ],
        )
    }

    fn create_bg2(&self) -> BindGroup {
        self.build_bg(
            "BG2",
            2,
            &[
                self.buffers.weights_buffer.as_ref().unwrap(),
                self.buffers.y_smooth_buffer.as_ref().unwrap(),
                self.buffers.residuals_buffer.as_ref().unwrap(),
                self.buffers.y_prev_buffer.as_ref().unwrap(),
                self.buffers.conf_lower_buffer.as_ref().unwrap(),
                self.buffers.conf_upper_buffer.as_ref().unwrap(),
                self.buffers.pred_lower_buffer.as_ref().unwrap(),
                self.buffers.pred_upper_buffer.as_ref().unwrap(),
            ],
        )
    }

    fn create_bg3(&self, median_buffer_override: Option<&Buffer>) -> BindGroup {
        let label = if median_buffer_override.is_some() {
            "BG3_Median"
        } else {
            "BG3"
        };
        let binding1 = median_buffer_override
            .unwrap_or_else(|| self.buffers.reduction_buffer.as_ref().unwrap());

        self.build_bg(
            label,
            3,
            &[
                self.buffers.w_config_buffer.as_ref().unwrap(),
                binding1,
                self.buffers.std_errors_buffer.as_ref().unwrap(),
                self.buffers.histogram_buffer.as_ref().unwrap(),
                self.buffers.global_max_diff_buffer.as_ref().unwrap(),
                self.buffers.scan_block_sums_buffer.as_ref().unwrap(),
                self.buffers.scan_indices_buffer.as_ref().unwrap(),
                self.buffers.workgroup_histograms_buffer.as_ref().unwrap(),
            ],
        )
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

        macro_rules! ensure {
            ($label:expr, $buf:expr, $size:expr, $usage:expr) => {
                bg_needs_update |=
                    GpuBuffers::ensure_capacity(&self.device, $label, $buf, $size, $usage);
            };
        }

        // Group 0: Config (Uniform)
        ensure!(
            "Config",
            &mut self.buffers.config_buffer,
            size_of::<GpuConfig>() as u64,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST
        );
        self.queue.write_buffer(
            self.buffers.config_buffer.as_ref().unwrap(),
            0,
            cast_slice(&[config]),
        );

        // Check for padding offset
        let pad_len = config.pad_len as u64;
        let offset = pad_len * 4;

        // Group 0: X, Y, Anchors, AnchorOutput
        ensure!(
            "X",
            &mut self.buffers.x_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );
        self.queue.write_buffer(
            self.buffers.x_buffer.as_ref().unwrap(),
            offset,
            cast_slice(x),
        );

        ensure!(
            "Y",
            &mut self.buffers.y_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );

        ensure!(
            "IndirectArgs",
            &mut self.buffers.indirect_buffer,
            256,
            BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC
        );
        self.queue.write_buffer(
            self.buffers.y_buffer.as_ref().unwrap(),
            offset,
            cast_slice(y),
        );

        ensure!(
            "AnchorIndices",
            &mut self.buffers.anchor_indices_buffer,
            anchor_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );
        ensure!(
            "AnchorOutput",
            &mut self.buffers.anchor_output_buffer,
            anchor_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );

        if bg_needs_update || self.bg0_data.is_none() {
            self.bg0_data = Some(self.create_bg0(
                self.buffers.x_buffer.as_ref().unwrap(),
                self.buffers.y_buffer.as_ref().unwrap(),
            ));
        }

        // Group 1: Interval Map
        bg_needs_update = false;
        let interval_bytes = (n_padded as usize * 8) as u64;
        ensure!(
            "IntervalMap",
            &mut self.buffers.interval_map_buffer,
            interval_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );

        if bg_needs_update || self.bg1_topo.is_none() {
            // Ensure test buffers are initialized (even if empty) to satisfy layout
            self.ensure_bg1_dummy_buffers();

            self.bg1_topo = Some(self.create_bg1());
        }

        // Group 2: State
        bg_needs_update = false;
        ensure!(
            "RobustnessWeights",
            &mut self.buffers.weights_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC
        );
        ensure!(
            "YSmooth",
            &mut self.buffers.y_smooth_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );
        ensure!(
            "Residuals",
            &mut self.buffers.residuals_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );
        ensure!(
            "YPrev",
            &mut self.buffers.y_prev_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );
        ensure!(
            "ConfLower",
            &mut self.buffers.conf_lower_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );
        ensure!(
            "ConfUpper",
            &mut self.buffers.conf_upper_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );
        ensure!(
            "PredLower",
            &mut self.buffers.pred_lower_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );
        ensure!(
            "PredUpper",
            &mut self.buffers.pred_upper_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );

        if bg_needs_update || self.bg2_state.is_none() {
            self.bg2_state = Some(self.create_bg2());
        }

        // Group 3: Aux
        bg_needs_update = false;
        ensure!(
            "WConfig",
            &mut self.buffers.w_config_buffer,
            size_of::<WeightConfig>() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );

        // Reduction buffer sized for 1M elements (4MB) to support radix sort paging
        let reduction_size = ((1024 * 1024) as usize * 4) as u64;
        ensure!(
            "Reduction",
            &mut self.buffers.reduction_buffer,
            reduction_size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );
        ensure!(
            "StdErrors",
            &mut self.buffers.std_errors_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );
        ensure!(
            "Histogram",
            &mut self.buffers.histogram_buffer,
            1024,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
        );

        // Global Max Diff Buffer (Atomic u32)
        ensure!(
            "Global Max Diff",
            &mut self.buffers.global_max_diff_buffer,
            4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST
        );
        ensure!(
            "ScanBlockSums",
            &mut self.buffers.scan_block_sums_buffer,
            (n_padded / 256 * 4 + 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST
        );
        ensure!(
            "ScanIndices",
            &mut self.buffers.scan_indices_buffer,
            n_bytes_padded,
            BufferUsages::STORAGE | BufferUsages::COPY_DST
        );
        ensure!(
            "WorkgroupHistograms",
            &mut self.buffers.workgroup_histograms_buffer,
            (n_padded.div_ceil(256) as u64 * 256 * 4).max(1024),
            BufferUsages::STORAGE | BufferUsages::COPY_DST
        );

        if bg_needs_update || self.bg3_aux.is_none() {
            self.bg3_aux = Some(self.create_bg3(None));
        }

        bg_needs_update |= GpuBuffers::ensure_capacity(
            &self.device,
            "Median",
            &mut self.buffers.median_buffer,
            reduction_size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );

        if bg_needs_update || self.bg3_median.is_none() {
            self.bg3_median =
                Some(self.create_bg3(Some(self.buffers.median_buffer.as_ref().unwrap())));
        }

        // Initialize WeightConfig
        self.queue.write_buffer(
            self.buffers.w_config_buffer.as_ref().unwrap(),
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
                _pad: 0,
            }),
        );

        // Staging
        let staging_size = n_bytes_padded.max(1024); // Ensure enough space for small downloads or median
        GpuBuffers::ensure_capacity(
            &self.device,
            "Staging",
            &mut self.buffers.staging_buffer,
            staging_size,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        self.n = n_padded;
        self.orig_n = orig_n;
        self.num_anchors = max_anchors; // Will be updated after anchor selection kernel runs

        // Run padding kernel if pad_len > 0
        if config.pad_len > 0 {
            {
                let mut encoder = self
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor {
                        label: Some("Boundary Padding"),
                    });
                self.record_pipeline(
                    &mut encoder,
                    &self.pipelines.pad_pipeline,
                    DispatchMode::Direct(config.pad_len.div_ceil(256)),
                );
                self.queue.submit(Some(encoder.finish()));
            }
            let _ = self.device.poll(PollType::Wait);
        }
    }

    pub fn record_full_scan(&self, encoder: &mut CommandEncoder, count: u32, bg3: &BindGroup) {
        self.record_pipeline_with_bg3(
            encoder,
            &self.pipelines.scan_block_pipeline,
            DispatchMode::Direct(count.div_ceil(256)),
            bg3,
        );
        self.record_pipeline_with_bg3(
            encoder,
            &self.pipelines.scan_aux_pipeline,
            DispatchMode::Direct(1),
            bg3,
        );
        self.record_pipeline_with_bg3(
            encoder,
            &self.pipelines.scan_add_base_pipeline,
            DispatchMode::Direct(count.div_ceil(256)),
            bg3,
        );
    }

    pub fn record_compact(
        &self,
        encoder: &mut CommandEncoder,
        count: u32,
        flags_pipeline: &ComputePipeline,
        compact_pipeline: &ComputePipeline,
        bg0: Option<&BindGroup>,
    ) {
        let actual_bg0 = bg0.unwrap_or(self.bg0_data.as_ref().unwrap());

        // 1. Prepare Flags
        self.record_pipeline_with_custom_bgs(
            encoder,
            flags_pipeline,
            DispatchMode::Direct(count.div_ceil(256)),
            actual_bg0,
            self.bg3_aux.as_ref().unwrap(),
        );

        // 2. Scan
        self.record_full_scan(encoder, count, self.bg3_aux.as_ref().unwrap());

        // 3. Compact
        self.record_pipeline_with_custom_bgs(
            encoder,
            compact_pipeline,
            DispatchMode::Direct(count.div_ceil(256)),
            actual_bg0,
            self.bg3_aux.as_ref().unwrap(),
        );
    }

    pub fn record_cv_partition(&self, encoder: &mut CommandEncoder, n_train: u32, n_test: u32) {
        let n_full = n_train + n_test;

        // 1. Prepare mask
        self.record_pipeline(
            encoder,
            &self.pipelines.cv_prepare_mask_pipeline,
            DispatchMode::Direct(n_full.div_ceil(256)),
        );

        // 2. Mark test indices
        self.record_pipeline(
            encoder,
            &self.pipelines.cv_mark_test_indices_pipeline,
            DispatchMode::Direct(n_test.div_ceil(256)),
        );

        // 3. Compact Training Set (Writes to X/Y in BG0)
        self.record_compact(
            encoder,
            n_full,
            &self.pipelines.cv_prepare_compact_flags_train_pipeline,
            &self.pipelines.cv_compact_training_pipeline,
            None, // Uses bg0_data
        );

        // 4. Compact Test Set (Writes to X_Test/Y_Test via bg0_test)
        self.record_compact(
            encoder,
            n_full,
            &self.pipelines.cv_prepare_compact_flags_test_pipeline,
            &self.pipelines.cv_compact_test_pipeline,
            self.bg0_test.as_ref(),
        );
    }

    pub fn record_cv_global_sort(&self, encoder: &mut CommandEncoder) {
        // Redraw data from global to buffers, sort it, then copy back to global

        // Mark all as training (copy global -> x/y)
        self.record_pipeline(
            encoder,
            &self.pipelines.cv_prepare_mask_pipeline,
            DispatchMode::Direct(self.n.div_ceil(256)),
        );

        self.record_compact(
            encoder,
            self.n,
            &self.pipelines.cv_prepare_compact_flags_train_pipeline,
            &self.pipelines.cv_compact_training_pipeline,
            None,
        );

        // Standard radix sort on x/y
        self.record_sort_input(encoder);

        // Copy back sorted x/y to global
        let pad_offset = (self.n - self.orig_n) as u64 / 2 * 4;
        encoder.copy_buffer_to_buffer(
            self.buffers.x_buffer.as_ref().unwrap(),
            pad_offset,
            self.buffers.x_global_buffer.as_ref().unwrap(),
            0,
            self.orig_n as u64 * 4,
        );
        encoder.copy_buffer_to_buffer(
            self.buffers.y_buffer.as_ref().unwrap(),
            pad_offset,
            self.buffers.y_global_buffer.as_ref().unwrap(),
            0,
            self.orig_n as u64 * 4,
        );
    }

    pub fn record_shuffle_indices(&self, encoder: &mut CommandEncoder) {
        // 1. Initialize Shuffle Data (X=Keys, Y=Values)
        // Uses BG0 (with config including seed)
        self.record_pipeline(
            encoder,
            &self.pipelines.init_shuffle_pipeline,
            DispatchMode::Direct(self.n.div_ceil(256)),
        );

        // 2. Sort X/Y (X=Random, Y=Indices)
        // This sorts 'x_buffer' and permutes 'y_buffer'
        self.record_sort_input(encoder);

        // 3. Copy Y (shuffled indices) to shuffled_indices_buffer
        encoder.copy_buffer_to_buffer(
            self.buffers.y_buffer.as_ref().unwrap(),
            0, // source offset
            self.buffers.shuffled_indices_buffer.as_ref().unwrap(),
            0, // dest offset
            self.n as u64 * 4,
        );
    }

    pub fn record_score_cv(&self, encoder: &mut CommandEncoder, n_test: u32) {
        self.record_pipeline(
            encoder,
            &self.pipelines.score_cv_points_pipeline,
            DispatchMode::Direct(n_test.div_ceil(256)),
        );
    }

    pub fn record_sum_sse_reduction(&self, encoder: &mut CommandEncoder, n_test: u32) {
        self.record_reduction(
            encoder,
            &self.pipelines.sum_sse_reduction_pipeline,
            &self.pipelines.finalize_sum_pipeline,
            n_test,
        );
    }

    pub fn record_sum_residuals_squared_reduction(&self, encoder: &mut CommandEncoder, n: u32) {
        self.record_reduction(
            encoder,
            &self.pipelines.sum_residuals_squared_pipeline,
            &self.pipelines.finalize_sum_pipeline,
            n,
        );
    }

    pub fn record_pipeline(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &ComputePipeline,
        dispatch: DispatchMode,
    ) {
        self.record_pipeline_with_bg3(encoder, pipeline, dispatch, self.bg3_aux.as_ref().unwrap());
    }

    pub fn record_pipeline_with_bg3(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &ComputePipeline,
        dispatch: DispatchMode,
        bg3: &BindGroup,
    ) {
        self.record_pipeline_with_custom_bgs(
            encoder,
            pipeline,
            dispatch,
            self.bg0_data.as_ref().unwrap(),
            bg3,
        );
    }

    pub fn record_pipeline_with_custom_bgs(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &ComputePipeline,
        dispatch: DispatchMode,
        bg0: &BindGroup,
        bg3: &BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg0, &[]);
        pass.set_bind_group(1, self.bg1_topo.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, self.bg2_state.as_ref().unwrap(), &[]);
        pass.set_bind_group(3, bg3, &[]);
        match dispatch {
            DispatchMode::Direct(n) => pass.dispatch_workgroups(n, 1, 1),
            DispatchMode::Indirect(offset) => pass.dispatch_workgroups_indirect(
                self.buffers.indirect_buffer.as_ref().unwrap(),
                offset,
            ),
        }
    }

    /// Record a control op (single-thread config update)
    fn record_control(&self, encoder: &mut CommandEncoder, op: ControlOp) {
        self.record_pipeline(
            encoder,
            &self.pipelines.control_pipelines[op as usize],
            DispatchMode::Direct(1),
        );
    }

    /// Record a reduction + finalization sequence
    fn record_reduction(
        &self,
        encoder: &mut CommandEncoder,
        reduce_pipeline: &ComputePipeline,
        finalize_pipeline: &ComputePipeline,
        n: u32,
    ) {
        self.record_pipeline(
            encoder,
            reduce_pipeline,
            DispatchMode::Direct(n.div_ceil(256)),
        );
        self.record_pipeline(encoder, finalize_pipeline, DispatchMode::Direct(1));
    }

    /// Helper to create encoder, record commands, and submit
    pub fn execute_commands(&self, f: impl FnOnce(&mut CommandEncoder)) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        f(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
    }

    /// Consolidates the prepare -> sort -> select -> update config flow for median-based scaling
    fn record_median_pass(
        &self,
        encoder: &mut CommandEncoder,
        prepare_op: PrepareOp,
        control_op: ControlOp,
    ) {
        let bg3_med = self.bg3_median.as_ref().unwrap();

        // 1. Prepare
        self.record_pipeline_with_bg3(
            encoder,
            &self.pipelines.prepare_pipelines[prepare_op as usize],
            DispatchMode::Direct(self.n.div_ceil(256)),
            bg3_med,
        );

        // 2. Sort
        self.record_radix_sort_passes(encoder, Some(bg3_med));

        // 3. Select Median
        self.record_pipeline_with_bg3(
            encoder,
            &self.pipelines.select_median_pipeline,
            DispatchMode::Direct(1),
            bg3_med,
        );

        // 4. Set Mode (Center or Scale)
        self.record_control(encoder, control_op);

        // 5. Update Config
        self.record_pipeline_with_bg3(
            encoder,
            &self.pipelines.update_scale_config_pipeline,
            DispatchMode::Direct(1),
            bg3_med,
        );
    }

    /// Helper for Mean fallback calculation used in MAR/MAD
    fn record_mean_fallback(&self, encoder: &mut CommandEncoder) {
        self.record_reduction(
            encoder,
            &self.pipelines.reduce_sum_abs_pipeline,
            &self.pipelines.finalize_sum_pipeline,
            self.n,
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DispatchMode {
    Direct(u32),
    Indirect(u64),
}

#[derive(Clone, Copy)]
pub struct RadixSortConfig<'a> {
    pub hist_pipeline: &'a ComputePipeline,
    pub hist_dispatch: DispatchMode,
    pub scan_histograms_pipeline: &'a ComputePipeline,
    pub prefix_dispatch: DispatchMode,
    pub scatter_pipeline: &'a ComputePipeline,
    pub scatter_dispatch: DispatchMode,
    pub copy_back_pipeline: &'a ComputePipeline,
    pub copy_dispatch: DispatchMode,
}

impl GpuExecutor {
    fn record_fit_pass(&self, encoder: &mut CommandEncoder, mode: DispatchMode, weight_fn: u32) {
        // 1. Fit Anchors (Indirect Dispatch - Slot 0)
        let pipeline = &self.pipelines.fit_pipelines[weight_fn as usize];
        self.record_pipeline(encoder, pipeline, DispatchMode::Indirect(0));
        // 2. Interpolate
        self.record_pipeline(encoder, &self.pipelines.interpolate_pipeline, mode);
    }

    fn record_scale_estimation(&self, encoder: &mut CommandEncoder) {
        // 1. Sum absolute residuals
        self.record_pipeline(
            encoder,
            &self.pipelines.reduce_sum_abs_pipeline,
            DispatchMode::Direct(self.n.div_ceil(256)),
        );
        // 2. Finalize scale (single thread)
        self.record_pipeline(
            encoder,
            &self.pipelines.finalize_scale_pipeline,
            DispatchMode::Direct(1),
        );
    }

    pub fn record_prepare_fit(&self, encoder: &mut CommandEncoder) {
        // 1. Select Anchors (Parallel)
        // 1a-e. Mark Candidates, Scan, Compact
        self.record_compact(
            encoder,
            self.n,
            &self.pipelines.mark_anchor_candidates_pipeline,
            &self.pipelines.compact_anchors_pipeline,
            None,
        );

        // 2. Compute intervals (parallel) - Slot 3
        self.record_pipeline(
            encoder,
            &self.pipelines.compute_intervals_pipeline,
            DispatchMode::Direct(self.n.div_ceil(256)),
        );

        // 3. Initialize Loop State
        self.record_control(encoder, ControlOp::InitState);

        // 4. Prepare First Next Pass (Slot 0 for Fit, Slot 3 for Interp/Weights, Slot 6 for Reduction)
        self.record_control(encoder, ControlOp::PrepareNext);

        // 5. Initialize weights to 1.0
        self.record_pipeline(
            encoder,
            &self.pipelines.init_weights_pipeline,
            DispatchMode::Direct(self.n.div_ceil(256)),
        );
    }

    pub fn record_fitting_loop(
        &self,
        encoder: &mut CommandEncoder,
        iterations: u32,
        scaling_method: ScalingMethod,
        weight_fn: u32,
    ) {
        for i in 0..=iterations {
            // Update Dispatch Args (Indirect Control)
            self.record_pipeline_with_bg3(
                encoder,
                &self.pipelines.update_dispatch_pipeline,
                DispatchMode::Direct(1), // Always run updater
                self.bg3_aux.as_ref().unwrap(),
            );

            if i > 0 {
                // Copy previous smoothed values to y_prev so convergence check
                // can compare y_smooth (current) vs y_prev (previous).
                let dst_buf = self.buffers.y_prev_buffer.as_ref().unwrap();
                let src_buf = self.buffers.y_smooth_buffer.as_ref().unwrap();
                encoder.copy_buffer_to_buffer(src_buf, 0, dst_buf, 0, src_buf.size());
            }

            // Use Indirect Slot 2 (24 bytes) for interpolate
            self.record_fit_pass(encoder, DispatchMode::Indirect(24), weight_fn);

            // 2. Record Convergence Check & Loop Control (Iteration > 0)
            if i > 0 {
                self.record_convergence_check(encoder); // Reducers use Slot 6
                self.record_control(encoder, ControlOp::FinalizeConv);
            }

            // 3. Record Scale estimation if iterations remain
            if i < iterations {
                self.record_robust_scale(encoder, scaling_method);

                // Use Indirect Slot 5 (60 bytes) for weight update
                self.record_update_weights(encoder, DispatchMode::Indirect(60));
            }

            // 4. Prepare for next iteration
            self.record_control(encoder, ControlOp::PrepareNext);
        }
    }

    pub fn update_config(
        &mut self,
        gpu_config: &GpuConfig,
        robustness_method: u32,
        scaling_method: u32,
    ) {
        self.n = gpu_config.n;
        self.orig_n = gpu_config.orig_n;

        // Update main config
        self.queue.write_buffer(
            self.buffers.config_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[*gpu_config]),
        );

        // Update WeightConfig status (state slot 0)
        let weight_config = WeightConfig {
            n: gpu_config.n,
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
            _pad: 0,
        };
        self.queue.write_buffer(
            self.buffers.w_config_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[weight_config]),
        );
    }

    fn ensure_bg1_dummy_buffers(&mut self) {
        let create_dummy = |label: &'static str| {
            self.device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        if self.buffers.interval_map_buffer.is_none() {
            self.buffers.interval_map_buffer = Some(create_dummy("IntervalMap_Dummy"));
        }
        if self.buffers.x_test_buffer.is_none() {
            self.buffers.x_test_buffer = Some(create_dummy("X_Test_Dummy"));
        }
        if self.buffers.y_test_buffer.is_none() {
            self.buffers.y_test_buffer = Some(create_dummy("Y_Test_Dummy"));
        }
        if self.buffers.test_errors_buffer.is_none() {
            self.buffers.test_errors_buffer = Some(create_dummy("TestErrors_Dummy"));
        }
        if self.buffers.x_global_buffer.is_none() {
            self.buffers.x_global_buffer = Some(create_dummy("X_Global_Dummy"));
        }
        if self.buffers.y_global_buffer.is_none() {
            self.buffers.y_global_buffer = Some(create_dummy("Y_Global_Dummy"));
        }
        if self.buffers.shuffled_indices_buffer.is_none() {
            self.buffers.shuffled_indices_buffer = Some(create_dummy("Shuffled_Indices_Dummy"));
        }
        if self.buffers.cv_test_mask_buffer.is_none() {
            self.buffers.cv_test_mask_buffer = Some(create_dummy("CV_Test_Mask_Dummy"));
        }
        if self.buffers.cv_results_buffer.is_none() {
            self.buffers.cv_results_buffer = Some(create_dummy("CV_Results_Dummy"));
        }
    }

    pub fn reset_cv_global_buffers(
        &mut self,
        x: &[f32],
        y: &[f32],
        shuffled_indices: &[u32],
        results_capacity: u64,
    ) {
        let n = x.len() as u64;
        let mut bg_needs_update = false;

        bg_needs_update |= GpuBuffers::ensure_capacity(
            &self.device,
            "X Global",
            &mut self.buffers.x_global_buffer,
            n * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        bg_needs_update |= GpuBuffers::ensure_capacity(
            &self.device,
            "Y Global",
            &mut self.buffers.y_global_buffer,
            n * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        bg_needs_update |= GpuBuffers::ensure_capacity(
            &self.device,
            "Shuffled Indices",
            &mut self.buffers.shuffled_indices_buffer,
            n * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        bg_needs_update |= GpuBuffers::ensure_capacity(
            &self.device,
            "CV Test Mask",
            &mut self.buffers.cv_test_mask_buffer,
            n * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        bg_needs_update |= GpuBuffers::ensure_capacity(
            &self.device,
            "CV Results",
            &mut self.buffers.cv_results_buffer,
            results_capacity,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        self.queue.write_buffer(
            self.buffers.x_global_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(x),
        );
        self.queue.write_buffer(
            self.buffers.y_global_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(y),
        );
        self.queue.write_buffer(
            self.buffers.shuffled_indices_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(shuffled_indices),
        );
        if bg_needs_update || self.bg1_topo.is_none() {
            // Ensure ALL buffers for BG1 are initialized (even if empty) to satisfy layout
            self.ensure_bg1_dummy_buffers();

            // Recreate BG1 which now contains CV global data at bindings 4-7
            self.bg1_topo = Some(self.create_bg1());
        }

        if bg_needs_update || self.bg0_test.is_none() {
            // Create BG0 variant that points to X_Test / Y_Test instead of main X/Y
            self.bg0_test = Some(self.create_bg0(
                self.buffers.x_test_buffer.as_ref().unwrap(),
                self.buffers.y_test_buffer.as_ref().unwrap(),
            ));
        }
    }

    pub fn record_generic_radix_sort(
        &self,
        encoder: &mut CommandEncoder,
        bg3: Option<&BindGroup>,
        config: RadixSortConfig,
    ) {
        // Reset Radix Pass
        self.record_control(encoder, ControlOp::ResetRadix);

        let bg = bg3.unwrap_or(self.bg3_aux.as_ref().unwrap());
        for _ in 0..4 {
            // 1. Clear Histogram
            self.record_pipeline_with_bg3(
                encoder,
                &self.pipelines.clear_histogram_pipeline,
                DispatchMode::Direct(512),
                bg,
            );

            // 2. Histogram
            self.record_pipeline_with_bg3(encoder, config.hist_pipeline, config.hist_dispatch, bg);

            // 3. Prefix Sum (Global and Workgroup)
            self.record_pipeline_with_bg3(
                encoder,
                config.scan_histograms_pipeline,
                config.prefix_dispatch,
                bg,
            );

            // 4. Scatter
            self.record_pipeline_with_bg3(
                encoder,
                config.scatter_pipeline,
                config.scatter_dispatch,
                bg,
            );

            // 5. Copy Back
            self.record_pipeline_with_bg3(
                encoder,
                config.copy_back_pipeline,
                config.copy_dispatch,
                bg,
            );

            // 6. Increment Radix Pass
            self.record_control(encoder, ControlOp::IncRadix);
        }
    }

    pub fn record_radix_sort_passes(&self, encoder: &mut CommandEncoder, bg3: Option<&BindGroup>) {
        let config = RadixSortConfig {
            hist_pipeline: &self.pipelines.radix_histogram_pipeline,
            hist_dispatch: DispatchMode::Indirect(24),
            scan_histograms_pipeline: &self.pipelines.radix_scan_histograms_pipeline,
            prefix_dispatch: DispatchMode::Direct(1),
            scatter_pipeline: &self.pipelines.radix_scatter_pipeline,
            scatter_dispatch: DispatchMode::Indirect(24),
            copy_back_pipeline: &self.pipelines.radix_copy_back_pipeline,
            copy_dispatch: DispatchMode::Indirect(24),
        };
        self.record_generic_radix_sort(encoder, bg3, config);
    }

    pub fn record_pad_data(&self, encoder: &mut CommandEncoder) {
        self.record_pipeline(
            encoder,
            &self.pipelines.pad_pipeline,
            DispatchMode::Direct(self.n.div_ceil(256)),
        );
    }

    fn record_update_weights(&self, encoder: &mut CommandEncoder, mode: DispatchMode) {
        // 3. Update Weights
        self.record_pipeline(encoder, &self.pipelines.weight_pipeline, mode);
    }

    fn record_convergence_check(&self, encoder: &mut CommandEncoder) {
        // Reduce max differences
        self.record_pipeline(
            encoder,
            &self.pipelines.reduce_max_diff_pipeline,
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
        encoder.copy_buffer_to_buffer(
            buf,
            offset,
            self.buffers.staging_buffer.as_ref().unwrap(),
            0,
            size,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.buffers.staging_buffer.as_ref().unwrap().slice(..size);
        let (tx, rx) = oneshot_channel();
        slice.map_async(MapMode::Read, move |v| tx.send(v).unwrap());
        let _ = self.device.poll(PollType::Wait);

        if let Some(Ok(())) = rx.receive().await {
            let data = slice.get_mapped_range();
            let ret = data.to_vec();
            drop(data);
            self.buffers.staging_buffer.as_ref().unwrap().unmap();
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
            &self.pipelines.select_median_pipeline,
            DispatchMode::Direct(1),
        );
        self.queue.submit(Some(encoder.finish()));

        // Download result from reduction_buffer[1048575]
        let result = self
            .download_buffer(
                self.buffers.reduction_buffer.as_ref().unwrap(),
                Some(4),
                Some(1048575 * 4),
            )
            .await
            .ok_or_else(|| "Failed to download median result".to_string())?;

        Ok(result[0])
    }

    /// Record commands to sort the input (x, y) based on x values.
    /// This uses a 4-pass radix sort and utilizes y_smooth/residuals as temporary buffers.
    pub fn record_sort_input(&self, encoder: &mut CommandEncoder) {
        let n_padded = self.n;

        let config = RadixSortConfig {
            hist_pipeline: &self.pipelines.sort_x_histogram_pipeline,
            hist_dispatch: DispatchMode::Direct(n_padded.div_ceil(256)),
            scan_histograms_pipeline: &self.pipelines.sort_x_scan_histograms_pipeline,
            prefix_dispatch: DispatchMode::Direct(1),
            scatter_pipeline: &self.pipelines.sort_x_scatter_pipeline,
            scatter_dispatch: DispatchMode::Direct(n_padded.div_ceil(256)),
            copy_back_pipeline: &self.pipelines.sort_x_copy_back_pipeline,
            copy_dispatch: DispatchMode::Direct(n_padded.div_ceil(256)),
        };

        // Use the generic radix sort implementation
        self.record_generic_radix_sort(encoder, None, config);
    }

    fn record_robust_scale(&self, encoder: &mut CommandEncoder, scaling_method: ScalingMethod) {
        match scaling_method {
            ScalingMethod::Mean => {
                self.record_scale_estimation(encoder);
            }
            ScalingMethod::MAR => {
                self.record_mean_fallback(encoder);
                self.record_median_pass(encoder, PrepareOp::MAR, ControlOp::SetScaleMode);
            }
            ScalingMethod::MAD => {
                self.record_mean_fallback(encoder);
                // Step 1: Center
                self.record_median_pass(encoder, PrepareOp::Signed, ControlOp::SetCenterMode);
                // Step 2: Scale
                self.record_median_pass(encoder, PrepareOp::MAD, ControlOp::SetScaleMode);
            }
        }
    }
}

// Compute standard errors for GPU results using GPU SE kernel
#[allow(clippy::type_complexity)]
fn compute_intervals_gpu<T>(
    exec: &mut GpuExecutor,
    config: &LowessConfig<T>,
    gpu_config: &mut GpuConfig,
) -> (
    Option<Vec<T>>,
    Option<Vec<T>>,
    Option<Vec<T>>,
    Option<Vec<T>>,
    Option<Vec<T>>,
)
where
    T: Float + Debug + Send + Sync + 'static,
{
    // Check if intervals requested
    let im = if let Some(im) = config.return_variance.as_ref() {
        im
    } else {
        return (None, None, None, None, None);
    };

    // Calculate parameters for intervals
    let z_score = IntervalMethod::approximate_z_score(im.level).unwrap_or(T::from(1.96).unwrap());

    // Update GPU config for interval pass (Part 1: CPU-known values)
    gpu_config.z_score = z_score.to_f32().unwrap_or(1.96);
    gpu_config.residual_sd = 0.0;
    gpu_config.has_conf = if im.confidence { 1 } else { 0 };
    gpu_config.has_pred = if im.prediction { 1 } else { 0 };

    exec.queue.write_buffer(
        exec.buffers.config_buffer.as_ref().unwrap(),
        0,
        cast_slice(&[*gpu_config]),
    );

    let mut encoder = exec
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Interval Bounds Calculation"),
        });

    // 1. Residual SSE Reduction
    exec.record_sum_residuals_squared_reduction(&mut encoder, exec.orig_n);

    // 2. Compute Residual SD (GPU)
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&exec.pipelines.compute_residual_sd_pipeline);
        pass.set_bind_group(0, exec.bg0_data.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, exec.bg1_topo.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, exec.bg2_state.as_ref().unwrap(), &[]);
        pass.set_bind_group(3, exec.bg3_aux.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // 3. Patch Config (Copy SD from Reduction[1] -> Config.residual_sd)
    // Offset in Config for residual_sd is 68 (17th u32/f32)
    // Offset in Reduction for [1] is 4
    encoder.copy_buffer_to_buffer(
        exec.buffers.reduction_buffer.as_ref().unwrap(),
        4,
        exec.buffers.config_buffer.as_ref().unwrap(),
        68,
        4,
    );

    // 4. Interval Passes
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

        // Standard Error Pass
        pass.set_pipeline(&exec.pipelines.se_pipeline);
        pass.set_bind_group(0, exec.bg0_data.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, exec.bg1_topo.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, exec.bg2_state.as_ref().unwrap(), &[]);
        pass.set_bind_group(3, exec.bg3_aux.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups_indirect(exec.buffers.indirect_buffer.as_ref().unwrap(), 12);

        // Interval Bounds Pass (if requested)
        if im.confidence || im.prediction {
            pass.set_pipeline(&exec.pipelines.interval_bounds_pipeline);
            pass.dispatch_workgroups_indirect(exec.buffers.indirect_buffer.as_ref().unwrap(), 12);
        }
    }

    exec.queue.submit(Some(encoder.finish()));
    let _ = exec.device.poll(PollType::Wait);

    // Download results with trimming
    let pad_len = (exec.n - exec.orig_n) / 2;
    let trim_offset = (pad_len as u64) * 4;
    let trim_size = (exec.orig_n as u64) * 4;

    let se_out = if im.se || im.confidence || im.prediction {
        let se_vals = block_on(exec.download_buffer(
            exec.buffers.std_errors_buffer.as_ref().unwrap(),
            Some(trim_size),
            Some(trim_offset),
        ))
        .unwrap();
        Some(cast_output_vec(se_vals))
    } else {
        None
    };

    let conf_lower = if im.confidence {
        let vals = block_on(exec.download_buffer(
            exec.buffers.conf_lower_buffer.as_ref().unwrap(),
            Some(trim_size),
            Some(trim_offset),
        ))
        .unwrap();
        Some(cast_output_vec(vals))
    } else {
        None
    };

    let conf_upper = if im.confidence {
        let vals = block_on(exec.download_buffer(
            exec.buffers.conf_upper_buffer.as_ref().unwrap(),
            Some(trim_size),
            Some(trim_offset),
        ))
        .unwrap();
        Some(cast_output_vec(vals))
    } else {
        None
    };

    let pred_lower = if im.prediction {
        let vals = block_on(exec.download_buffer(
            exec.buffers.pred_lower_buffer.as_ref().unwrap(),
            Some(trim_size),
            Some(trim_offset),
        ))
        .unwrap();
        Some(cast_output_vec(vals))
    } else {
        None
    };

    let pred_upper = if im.prediction {
        let vals = block_on(exec.download_buffer(
            exec.buffers.pred_upper_buffer.as_ref().unwrap(),
            Some(trim_size),
            Some(trim_offset),
        ))
        .unwrap();
        Some(cast_output_vec(vals))
    } else {
        None
    };

    (se_out, conf_lower, conf_upper, pred_lower, pred_upper)
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

        let total_n = orig_n; // Initial total_n before padding
        let window_size =
            (config.fraction.unwrap().to_f32().unwrap() * orig_n as f32).max(1.0) as usize;
        // Ensure window_size is at least 2 for meaningful calculations, and not larger than total_n
        let window_size = window_size.max(2usize).min(total_n);

        // Calculate padding (GPU-side padding)
        let pad_len = if config.boundary_policy == BoundaryPolicy::NoBoundary {
            0
        } else {
            (window_size / 2).min(orig_n - 1)
        };
        let total_n_padded = total_n + 2 * pad_len;

        // Convert original data to f32 (padding happens on GPU)
        let x_f32 = cast_input_slice(x);
        let y_f32 = cast_input_slice(y);

        // Calculate anchors based on PADDED range
        let mut delta = config.delta.to_f32().unwrap();

        // Auto-Delta Heuristic:
        // If delta is 0, default to 1% of the data range.
        if delta.abs() < 1e-12 && orig_n > 0 {
            // Find min/max robustly (input might be unsorted)
            let (min, max) = x_f32
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                    (min.min(val), max.max(val))
                });

            let range = max - min;
            if range > 0.0 {
                delta = range * 0.01;
            }
        }

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

        let mut gpu_config = GpuConfig {
            n: total_n_padded as u32,
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
            z_score: 0.0,
            has_conf: 0,
            has_pred: 0,
            residual_sd: 0.0,
            n_test: 0,
            seed: 0,
            has_se: if config.return_variance.is_some() {
                1
            } else {
                0
            },
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
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

        exec.reset_buffers(&x_f32, &y_f32, gpu_config, robustness_id, scaling_id);

        // Execute all GPU operations in a single command encoder (no synchronization)
        {
            let mut encoder = exec.device.create_command_encoder(&Default::default());

            // 1. Sort and pad input data
            exec.record_sort_input(&mut encoder);
            exec.record_pad_data(&mut encoder);

            // 2. Prepare for fitting (anchors, intervals, init)
            exec.record_prepare_fit(&mut encoder);

            exec.queue.submit(Some(encoder.finish()));
        }

        {
            let mut encoder = exec.device.create_command_encoder(&Default::default());
            // 3. Main fitting loop (all iterations)
            exec.record_fitting_loop(
                &mut encoder,
                config.iterations as u32,
                config.scaling_method,
                weight_fn_id,
            );

            // Submit everything at once - GPU executes without CPU intervention
            exec.queue.submit(Some(encoder.finish()));
        }

        let trim_offset = (pad_len as u64) * 4;
        let trim_size = (orig_n as u64) * 4;

        let y_res = block_on(exec.download_buffer(
            exec.buffers.y_smooth_buffer.as_ref().unwrap(),
            Some(trim_size),
            Some(trim_offset),
        ))
        .unwrap();

        let w_res = block_on(exec.download_buffer(
            exec.buffers.weights_buffer.as_ref().unwrap(),
            Some(trim_size),
            Some(trim_offset),
        ))
        .unwrap();

        let r_res = block_on(exec.download_buffer(
            exec.buffers.residuals_buffer.as_ref().unwrap(),
            Some(trim_size),
            Some(trim_offset),
        ))
        .unwrap();

        // Results are already trimmed by download_buffer
        let y_out: Vec<T> = cast_output_vec(y_res);
        let w_out: Vec<T> = cast_output_vec(w_res);
        let r_out: Vec<T> = cast_output_vec(r_res);

        // Compute standard errors/intervals if requested
        let (std_errors, conf_lower, conf_upper, pred_lower, pred_upper) =
            if config.return_variance.is_some() {
                compute_intervals_gpu(exec, config, &mut gpu_config)
            } else {
                (None, None, None, None, None)
            };

        let iterations_performed = block_on(exec.download_buffer_raw(
            exec.buffers.w_config_buffer.as_ref().unwrap(),
            Some(4),
            Some(32),
        ))
        .map(|raw| cast_slice::<u8, u32>(raw.as_slice())[0])
        .unwrap_or(0) as usize;

        Ok((
            y_out,
            std_errors,
            iterations_performed,
            w_out,
            Some(r_out),
            conf_lower,
            conf_upper,
            pred_lower,
            pred_upper,
        ))
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

        let mut guard = GLOBAL_EXECUTOR
            .lock()
            .map_err(|e| LowessError::RuntimeError(format!("GPU mutex poisoned: {}", e)))
            .unwrap();

        if guard.is_none() {
            match block_on(GpuExecutor::new()) {
                Ok(exec) => *guard = Some(exec),
                Err(_) => return (T::zero(), vec![T::zero(); fractions.len()]),
            }
        }
        let exec = guard.as_mut().unwrap();

        let n = x.len();
        let x_f32 = cast_input_slice(x);
        let y_f32 = cast_input_slice(y);

        // 1. Initial Buffer Setup (Ensures all bind groups are initialized)
        // Pre-allocate for worst-case padding (approximately n + window_size)
        let max_frac = fractions
            .iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(T::zero())
            .to_f32()
            .unwrap_or(0.0);
        let max_window = (max_frac * n as f32).max(2.0) as u32;
        let max_pad = (max_window / 2).min(n as u32 - 1);
        let max_n_padded = n as u32 + 2 * max_pad;

        let gpu_config = GpuConfig {
            n: max_n_padded,
            window_size: (n as f32 * 0.1) as u32, // Dummy
            weight_function: 0,
            zero_weight_fallback: 0,
            fraction: 0.1,
            delta: 0.0,
            median_threshold: 0.0,
            median_center: 0.0,
            is_absolute: 1,
            boundary_policy: 1,
            pad_len: 0,
            orig_n: n as u32,
            max_iterations: 1,
            tolerance: 1e-3,
            z_score: 1.96,
            has_conf: 0,
            has_pred: 0,
            residual_sd: 1.0,
            n_test: 0,
            seed: config.cv_seed.unwrap_or(12345) as u32,
            has_se: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        exec.reset_buffers(&x_f32, &y_f32, gpu_config, 0, 0);

        // Update to correct 'n' for global sort (while maintaining capacity)
        let mut sort_config = gpu_config;
        sort_config.n = n as u32;
        exec.update_config(&sort_config, 0, 0);
        exec.n = n as u32;

        // 2. Upload Global Data
        let results_size = (fractions.len() * 4) as u64;
        exec.reset_cv_global_buffers(&x_f32, &y_f32, &[], results_size);

        // 3. Shuffle Indices on GPU
        exec.execute_commands(|encoder| {
            exec.record_shuffle_indices(encoder);
        });

        // 4. Sort Global Data once on GPU
        exec.orig_n = n as u32;
        exec.execute_commands(|encoder| {
            exec.record_cv_global_sort(encoder);
        });

        // Common config parameters
        let delta = config.delta.to_f32().unwrap();
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

        let mut scores = vec![T::infinity(); fractions.len()];

        // Zero out result buffer first
        exec.execute_commands(|encoder| {
            encoder.clear_buffer(exec.buffers.cv_results_buffer.as_ref().unwrap(), 0, None);
        });

        for (frac_idx, &frac) in fractions.iter().enumerate() {
            let k = match method {
                CVKind::KFold(k) => k,
                CVKind::LOOCV => n,
            };

            if n < k || k < 2 {
                continue;
            }

            let fold_size = n / k;

            // We use 'radix_pass' in WeightConfig to pass the frac_index to the accumulate shader
            // reusing the existing config struct.
            let mut gpu_config = GpuConfig {
                n: 0,           // Updated per fold
                window_size: 0, // Updated per fold
                weight_function: weight_fn_id,
                zero_weight_fallback: fallback_id,
                fraction: frac.to_f32().unwrap(),
                delta,
                median_threshold: 0.0,
                median_center: 0.0,
                is_absolute: 0,
                boundary_policy: boundary_id,
                pad_len: 0, // Updated per fold
                orig_n: 0,  // Updated per fold
                max_iterations: config.iterations as u32,
                tolerance: config
                    .auto_convergence
                    .map(|t| t.to_f32().unwrap())
                    .unwrap_or(0.0),
                z_score: 0.0, // Overloaded for test_start offset
                has_conf: 0,
                has_pred: 0,
                residual_sd: 0.0,
                n_test: 0,
                seed: 0,
                has_se: 0,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };
            for fold in 0..k {
                let test_start = (fold * fold_size) as u32;
                let test_end = (if fold == k - 1 {
                    n
                } else {
                    (fold + 1) * fold_size
                }) as u32;
                let n_test = test_end - test_start;
                let n_train = (n as u32) - n_test;

                if n_train < 2 {
                    continue;
                }

                // Configure for this fold
                let window_size = (frac.to_f32().unwrap() * n_train as f32).max(1.0) as u32;
                let window_size = window_size.max(2).min(n_train);
                let pad_len = if config.boundary_policy == BoundaryPolicy::NoBoundary {
                    0
                } else {
                    (window_size / 2).min(n_train - 1)
                };
                let total_n_padded = n_train + 2 * pad_len;

                gpu_config.n = total_n_padded;
                gpu_config.window_size = window_size;
                gpu_config.pad_len = pad_len;
                gpu_config.orig_n = n_train;
                gpu_config.z_score = test_start as f32; // Overloaded for test_start offset
                gpu_config.n_test = n_test;

                exec.update_config(&gpu_config, robustness_id, scaling_id);

                let wc = WeightConfig {
                    n: total_n_padded,
                    scale: 0.0,
                    robustness_method: robustness_id,
                    scaling_method: scaling_id,
                    median_center: 0.0,
                    mean_abs: 0.0,
                    anchor_count: 0,
                    radix_pass: frac_idx as u32,
                    converged: 0,
                    iteration: 0,
                    update_mode: 0,
                    _pad: 0,
                };
                exec.queue.write_buffer(
                    exec.buffers.w_config_buffer.as_ref().unwrap(),
                    0,
                    bytes_of(&wc),
                );

                exec.execute_commands(|encoder| {
                    // 1. Partition & Pad
                    exec.record_cv_partition(encoder, n_train, n_test);
                    exec.record_pad_data(encoder);

                    // 2. Fit
                    exec.record_prepare_fit(encoder);
                    exec.record_fitting_loop(
                        encoder,
                        config.iterations as u32,
                        config.scaling_method,
                        weight_fn_id,
                    );

                    // 3. Score
                    exec.record_score_cv(encoder, n_test);
                    exec.record_sum_sse_reduction(encoder, n_test);

                    // 4. Accumulate result
                    exec.record_pipeline(
                        encoder,
                        &exec.pipelines.accumulate_score_pipeline,
                        DispatchMode::Direct(1),
                    );
                });
            }
        }

        let _ = exec.device.poll(PollType::Wait);

        // Download all results at once
        let raw_results = block_on(exec.download_buffer(
            exec.buffers.cv_results_buffer.as_ref().unwrap(),
            Some((fractions.len() * 4) as u64),
            None,
        ))
        .unwrap();

        // Post-process scores (RMSE average)
        for (i, &total_rmse) in raw_results.iter().enumerate() {
            let k = match method {
                CVKind::KFold(k) => k,
                CVKind::LOOCV => n,
            };

            let count = k as f32;

            if total_rmse > 0.0 {
                scores[i] = T::from(total_rmse / count).unwrap();
            } else {
                scores[i] = T::zero();
            }
        }

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

/// Helper to convert a generic float slice to f32, using zero-copy if possible.
fn cast_input_slice<'a, T>(slice: &'a [T]) -> Cow<'a, [f32]>
where
    T: Float + 'static,
{
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let ptr = slice.as_ptr() as *const f32;
        let s = unsafe { std::slice::from_raw_parts(ptr, slice.len()) };
        Cow::Borrowed(s)
    } else {
        Cow::Owned(slice.iter().map(|v| v.to_f32().unwrap()).collect())
    }
}

/// Helper to convert a f32 vector to a generic float vector, using zero-copy if possible.
fn cast_output_vec<T>(vec: Vec<f32>) -> Vec<T>
where
    T: Float + 'static,
{
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let v_any = Box::new(vec) as Box<dyn Any>;
        *v_any.downcast::<Vec<T>>().expect("TypeId mismatch")
    } else {
        vec.into_iter().map(|v| T::from(v).unwrap()).collect()
    }
}
