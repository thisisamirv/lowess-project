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
use std::mem::{size_of, swap};
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

struct WeightConfig {
    n: u32,
    scale: f32,
    robustness_method: u32,
    scaling_method: u32,
}

// Group 0: Constants & Input Data
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> anchor_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> anchor_output: array<f32>;

// Group 1: Topology
@group(1) @binding(0) var<storage, read> interval_map: array<u32>;

// Group 2: State (Weights, Output, Residuals)
@group(2) @binding(0) var<storage, read_write> robustness_weights: array<f32>;
@group(2) @binding(1) var<storage, read_write> y_smooth: array<f32>;
@group(2) @binding(2) var<storage, read_write> residuals: array<f32>;
@group(2) @binding(3) var<storage, read_write> y_prev: array<f32>;

// Group 3: Aux (Reduction & Weight Config)
@group(3) @binding(0) var<storage, read_write> w_config: WeightConfig;
@group(3) @binding(1) var<storage, read_write> reduction: array<f32>;
@group(3) @binding(2) var<storage, read_write> std_errors: array<f32>;

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
    let num_anchors = arrayLength(&anchor_indices);

    var left = 0u;
    var right = 0u;
    var x_i = 0.0;
    var i = 0u;
    var valid_anchor = false;

    if (anchor_id < num_anchors) {
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

        let d_max_val = max(d_max, 1e-9);
        for (var k = left; k <= right; k++) {
            let xj = x[k];
            let yj = y[k];
            let rw = robustness_weights[k];
            
            let dist = abs(xj - x_i);
            let u = dist / d_max_val;
            
            sum_y_window += yj;
            
            if (u < 1.0) {
                var kernel_w = 1.0;
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
            let mean_x = sum_wx / sum_w;
            let mean_y = sum_wy / sum_w;
            let variance = (sum_wxx / sum_w) - (mean_x * mean_x);
            
            let ABS_TOL: f32 = 1e-7;
            let REL_TOL: f32 = 1e-7 * d_max_val * d_max_val;
            let tol = max(ABS_TOL, REL_TOL);

            if (variance <= tol) {
                anchor_output[anchor_id] = mean_y;
            } else {
                let covariance = (sum_wxy / sum_w) - (mean_x * mean_y);
                let slope = covariance / variance;
                let intercept = mean_y - slope * mean_x;
                
                if (intercept == intercept && abs(intercept) < 1e15) {
                    anchor_output[anchor_id] = intercept;
                } else {
                    anchor_output[anchor_id] = mean_y;
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
    var scale_est = total_sum / f32(config.n);

    if (w_config.scaling_method != SCALING_MEAN) {
        scale_est = scale_est * 0.845347;
    }

    w_config.scale = max(scale_est, 1e-10);
}

@compute @workgroup_size(1)
fn finalize_sum() {
    var total_sum = 0.0;
    let num_workgroups = (config.n + 255u) / 256u;
    for (var i = 0u; i < num_workgroups; i = i + 1u) {
        total_sum += reduction[i];
    }
    reduction[0] = total_sum;
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
"#;

const ROBUSTNESS_BISQUARE: u32 = 0;
const ROBUSTNESS_HUBER: u32 = 1;
const ROBUSTNESS_TALWAR: u32 = 2;

const SCALING_MAD: u32 = 0;
const SCALING_MAR: u32 = 1;
const SCALING_MEAN: u32 = 2;

static GLOBAL_EXECUTOR: Mutex<Option<GpuExecutor>> = Mutex::new(None);

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuConfig {
    n: u32,
    window_size: u32,
    weight_function: u32,
    zero_weight_fallback: u32,
    fraction: f32,
    delta: f32,
    median_threshold: f32,
    median_center: f32,
    is_absolute: u32,
    boundary_policy: u32,
    pad_len: u32,
    orig_n: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WeightConfig {
    n: u32,
    scale: f32,
    robustness_method: u32,
    scaling_method: u32,
}

struct GpuExecutor {
    device: Device,
    queue: Queue,

    // Pipelines
    fit_pipeline: ComputePipeline,
    interpolate_pipeline: ComputePipeline,
    weight_pipeline: ComputePipeline,
    reduce_max_diff_pipeline: ComputePipeline,
    reduce_sum_abs_pipeline: ComputePipeline,
    finalize_scale_pipeline: ComputePipeline,
    se_pipeline: ComputePipeline,
    pad_pipeline: ComputePipeline,

    // Buffers - Group 0
    config_buffer: Option<Buffer>,
    x_buffer: Option<Buffer>,
    y_buffer: Option<Buffer>,
    anchor_indices_buffer: Option<Buffer>,
    anchor_output_buffer: Option<Buffer>,

    // Buffers - Group 1
    interval_map_buffer: Option<Buffer>,

    // Buffers - Group 2
    weights_buffer: Option<Buffer>,
    y_smooth_buffer: Option<Buffer>,
    y_prev_buffer: Option<Buffer>, // For auto-convergence checking
    residuals_buffer: Option<Buffer>,

    // Buffers - Group 3
    w_config_buffer: Option<Buffer>,
    reduction_buffer: Option<Buffer>,
    std_errors_buffer: Option<Buffer>,

    // Staging
    staging_buffer: Option<Buffer>,

    // Bind Groups
    bg0_data: Option<BindGroup>,
    bg1_topo: Option<BindGroup>,
    bg2_state: Option<BindGroup>,
    bg3_aux: Option<BindGroup>,

    n: u32,
    num_anchors: u32,
}
impl GpuExecutor {
    async fn new() -> Result<Self, String> {
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
                        ty: BufferBindingType::Storage { read_only: true },
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
            ],
        });

        let bind_group_layout_1 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BG1 Topo"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
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
            se_pipeline: create_pipeline("compute_se"),
            pad_pipeline: create_pipeline("pad_data"),
            device,
            queue,
            config_buffer: None,
            x_buffer: None,
            y_buffer: None,
            anchor_indices_buffer: None,
            anchor_output_buffer: None,
            interval_map_buffer: None,
            weights_buffer: None,
            y_smooth_buffer: None,
            y_prev_buffer: None,
            residuals_buffer: None,
            w_config_buffer: None,
            reduction_buffer: None,
            std_errors_buffer: None,
            staging_buffer: None,
            bg0_data: None,
            bg1_topo: None,
            bg2_state: None,
            bg3_aux: None,
            n: 0,
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
    fn reset_buffers(
        &mut self,
        x: &[f32],
        y: &[f32],
        anchors: &[u32],
        intervals: &[u32],
        rob_w: &[f32],
        config: GpuConfig,
        robustness_method: u32,
        scaling_method: u32,
    ) {
        let n_padded = config.n;
        let num_anchors = anchors.len() as u32;
        let n_bytes_padded = (n_padded as usize * 4) as u64;
        let anchor_bytes = (num_anchors as usize * 4) as u64;

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
        self.queue.write_buffer(
            self.anchor_indices_buffer.as_ref().unwrap(),
            0,
            cast_slice(anchors),
        );

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
        self.queue.write_buffer(
            self.interval_map_buffer.as_ref().unwrap(),
            0,
            cast_slice(intervals),
        );

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
        self.queue
            .write_buffer(self.weights_buffer.as_ref().unwrap(), 0, cast_slice(rob_w));

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

        let reduction_size = 16.max((n_padded.div_ceil(256) as usize * 8) as u64);
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

        if bg_needs_update || self.bg3_aux.is_none() {
            self.bg3_aux = Some(self.device.create_bind_group(&BindGroupDescriptor {
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
                ],
            }));
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
            }),
        );

        // Staging
        Self::ensure_buffer_capacity(
            &self.device,
            "Staging",
            &mut self.staging_buffer,
            n_bytes_padded,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        self.n = n_padded;
        self.num_anchors = num_anchors;

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
                pass.dispatch_workgroups(config.pad_len.div_ceil(256), 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
            let _ = self.device.poll(PollType::Wait);
        }
    }

    fn swap_y_buffers(&mut self) {
        swap(&mut self.y_smooth_buffer, &mut self.y_prev_buffer);
        // Re-create BG2 to reflect the swap
        self.bg2_state = Some(self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("BG2 Swapped"),
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

    fn record_pipeline(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &ComputePipeline,
        dispatch_size: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, self.bg0_data.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.bg1_topo.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, self.bg2_state.as_ref().unwrap(), &[]);
        pass.set_bind_group(3, self.bg3_aux.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(dispatch_size, 1, 1);
    }

    fn record_fit_pass(&self, encoder: &mut CommandEncoder) {
        // 1. Fit Anchors
        self.record_pipeline(encoder, &self.fit_pipeline, self.num_anchors.div_ceil(64));
        // 2. Interpolate
        self.record_pipeline(encoder, &self.interpolate_pipeline, self.n.div_ceil(64));
    }

    fn record_scale_estimation(&self, encoder: &mut CommandEncoder) {
        // 1. Sum absolute residuals
        self.record_pipeline(encoder, &self.reduce_sum_abs_pipeline, self.n.div_ceil(256));
        // 2. Finalize scale (single thread)
        self.record_pipeline(encoder, &self.finalize_scale_pipeline, 1);
    }

    fn record_update_weights(&self, encoder: &mut CommandEncoder) {
        // 3. Update Weights
        self.record_pipeline(encoder, &self.weight_pipeline, self.n.div_ceil(64));
    }

    fn record_convergence_check(&self, encoder: &mut CommandEncoder) {
        // Reduce max differences
        self.record_pipeline(
            encoder,
            &self.reduce_max_diff_pipeline,
            self.n.div_ceil(256),
        );
    }

    async fn download_buffer(&self, buf: &Buffer, size_override: Option<u64>) -> Option<Vec<f32>> {
        let size = size_override.unwrap_or((self.n as usize * 4) as u64);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, self.staging_buffer.as_ref().unwrap(), 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.as_ref().unwrap().slice(..size);
        let (tx, rx) = oneshot_channel();
        slice.map_async(MapMode::Read, move |v| tx.send(v).unwrap());
        let _ = self.device.poll(PollType::Wait);

        if let Some(Ok(())) = rx.receive().await {
            let data = slice.get_mapped_range();
            let ret = cast_slice(&data).to_vec();
            drop(data);
            self.staging_buffer.as_ref().unwrap().unmap();
            Some(ret)
        } else {
            None
        }
    }

    async fn compute_robust_scale(
        &mut self,
        scaling_method: ScalingMethod,
    ) -> Result<Option<f32>, String> {
        // GPU-side scale estimation for Mean method (fast path)
        if scaling_method == ScalingMethod::Mean {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.record_scale_estimation(&mut encoder);
            self.queue.submit(Some(encoder.finish()));
            return Ok(None);
        }

        // For MAD/MAR, use host-based computation for accuracy and stability
        let residuals = self
            .download_buffer(self.residuals_buffer.as_ref().unwrap(), None)
            .await
            .ok_or_else(|| "Failed to download residuals".to_string())?;

        let mut residuals_mut: Vec<f32> = residuals;
        let scale = scaling_method.compute(&mut residuals_mut);
        Ok(Some(scale))
    }
}

// Check if convergence has been reached by comparing current and previous smoothed values
async fn check_convergence_gpu_internal(
    exec: &mut GpuExecutor,
    tolerance: f32,
) -> Result<bool, &'static str> {
    let num_workgroups = exec.n.div_ceil(256);
    let reduction_bytes = (num_workgroups as usize * 4) as u64;
    let reduction_vals = exec
        .download_buffer(
            exec.reduction_buffer.as_ref().unwrap(),
            Some(reduction_bytes),
        )
        .await
        .ok_or("Failed to download reduction results")?;

    // Final reduction on host
    let max_diff = reduction_vals.iter().fold(0.0f32, |m, &v| m.max(v));

    Ok(max_diff < tolerance)
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
        pass.dispatch_workgroups(exec.n.div_ceil(64), 1, 1);
    }

    exec.queue.submit(Some(encoder.finish()));
    let _ = exec.device.poll(PollType::Wait);

    // Download standard errors
    let se_vals = block_on(exec.download_buffer(exec.std_errors_buffer.as_ref().unwrap(), None))?;

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

        // Compute Anchors & Intervals on original coordinates, but indices shifted by pad_len
        let mut anchors = Vec::with_capacity(orig_n / 10 + 2);
        let mut intervals = vec![0u32; total_n];

        let mut last_idx = 0;
        anchors.push(pad_len as u32);
        let mut current_anchor_idx = 0;

        for i in 0..orig_n {
            if i > 0 && (x[i] - x[last_idx]).to_f32().unwrap() > delta {
                last_idx = i;
                anchors.push((i + pad_len) as u32);
                current_anchor_idx += 1;
            }
            intervals[i + pad_len] = current_anchor_idx;
        }

        // Pad intervals
        intervals[0..pad_len].fill(0);
        intervals[pad_len + orig_n..total_n].fill(current_anchor_idx);

        // Ensure last original point is an anchor
        if *anchors.last().unwrap() != (pad_len + orig_n - 1) as u32 {
            anchors.push((pad_len + orig_n - 1) as u32);
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

        exec.reset_buffers(
            &x_f32,
            &y_f32,
            &anchors,
            &intervals,
            &vec![1.0; total_n],
            gpu_config,
            robustness_id,
            scaling_id,
        );

        let mut iterations_performed = 0;
        let tolerance = config.auto_convergence.map(|t| t.to_f32().unwrap());

        for i in 0..=config.iterations {
            iterations_performed = i;

            let mut encoder = exec
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some(format!("LOWESS Pass {}", i).as_str()),
                });

            // 1. Record Fit
            if tolerance.is_some() && i > 0 {
                exec.swap_y_buffers();
            }
            exec.record_fit_pass(&mut encoder);

            // 2. Record Convergence Reduction (if needed)
            if let Some(_) = tolerance
                && i > 0
            {
                exec.record_convergence_check(&mut encoder);
            }

            // 3. Record Scale & Weight Update (if more iterations remain)
            let mut needs_host_scale = false;
            if i < config.iterations {
                if config.scaling_method == ScalingMethod::Mean {
                    exec.record_scale_estimation(&mut encoder);
                    exec.record_update_weights(&mut encoder);
                } else {
                    needs_host_scale = true;
                }
            }

            exec.queue.submit(Some(encoder.finish()));

            // Handle host-side scale estimation if needed (breaks batching)
            if needs_host_scale {
                let scale_opt = block_on(exec.compute_robust_scale(config.scaling_method))
                    .map_err(LowessError::RuntimeError)?;

                if let Some(scale) = scale_opt {
                    let w_config_host = WeightConfig {
                        n: total_n as u32,
                        scale,
                        robustness_method: robustness_id,
                        scaling_method: scaling_id,
                    };
                    exec.queue.write_buffer(
                        exec.w_config_buffer.as_ref().unwrap(),
                        0,
                        bytes_of(&w_config_host),
                    );

                    // Finalize weights for this iteration
                    let mut encoder_w = exec.device.create_command_encoder(&Default::default());
                    exec.record_update_weights(&mut encoder_w);
                    exec.queue.submit(Some(encoder_w.finish()));
                }
            }

            // Check convergence
            if let Some(tol) = tolerance
                && i > 0
                && block_on(check_convergence_gpu_internal(exec, tol)).unwrap_or(false)
            {
                break;
            }
        }

        let y_res =
            block_on(exec.download_buffer(exec.y_smooth_buffer.as_ref().unwrap(), None)).unwrap();
        let w_res =
            block_on(exec.download_buffer(exec.weights_buffer.as_ref().unwrap(), None)).unwrap();

        // Trim results to original size (remove padding)
        let y_trimmed = if pad_len > 0 {
            y_res[pad_len..pad_len + orig_n].to_vec()
        } else {
            y_res
        };
        let w_trimmed = if pad_len > 0 {
            w_res[pad_len..pad_len + orig_n].to_vec()
        } else {
            w_res
        };

        let y_out: Vec<T> = y_trimmed.into_iter().map(|v| T::from(v).unwrap()).collect();
        let w_out: Vec<T> = w_trimmed.into_iter().map(|v| T::from(v).unwrap()).collect();

        // Compute standard errors/intervals if requested
        let std_errors = if config.return_variance.is_some() {
            compute_intervals_gpu(exec, config).map(|v| {
                if pad_len > 0 {
                    v[pad_len..pad_len + orig_n].to_vec()
                } else {
                    v
                }
            })
        } else {
            None
        };

        Ok((y_out, std_errors, iterations_performed, w_out))
    }
    #[cfg(not(feature = "gpu"))]
    {
        unimplemented!("GPU feature disabled")
    }
}
