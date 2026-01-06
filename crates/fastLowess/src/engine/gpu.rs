//! GPU-accelerated execution engine for LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the GPU-accelerated smoothing function for LOWESS
//! operations. It leverages `wgpu` to execute local regression fits in parallel
//! on the GPU, providing maximum throughput for large-scale data processing.
//!
//! ## Optimizations
//!
//! * **Delta Interpolation**: Uses "anchors" (subset of points) for fitting,
//!   then interpolates remaining points. This reduces complexity from O(N^2)
//!   to O(Anchors * Width + N).
//!

use bytemuck::{Pod, Zeroable};
use num_traits::Float;
use std::fmt::Debug;

// Export dependencies from lowess crate
use lowess::internals::engine::executor::LowessConfig;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Instance, InstanceDescriptor, MapMode, PipelineLayoutDescriptor, PollType, Queue,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

// -----------------------------------------------------------------------------
// Shader Source (WGSL)
// -----------------------------------------------------------------------------
const SHADER_SOURCE: &str = r#"
struct Config {
    n: u32,
    window_size: u32,
    weight_function: u32, // Unused in this simplified shader (always Tricube)
    zero_weight_fallback: u32, // Unused
    fraction: f32,
    delta: f32,
}

struct WeightConfig {
    n: u32,
    scale: f32,
}

// Group 0: Constants & Input Data
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read> anchor_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> anchor_output: array<f32>;

// Group 1: Topology
@group(1) @binding(0) var<storage, read> interval_map: array<u32>;

// Group 2: State (Weights, Output, Residuals)
@group(2) @binding(0) var<storage, read_write> robustness_weights: array<f32>;
@group(2) @binding(1) var<storage, read_write> y_smooth: array<f32>;
@group(2) @binding(2) var<storage, read_write> residuals: array<f32>;

// Group 3: Aux (Reduction & Weight Config)
@group(3) @binding(0) var<storage, read_write> w_config: WeightConfig;
@group(3) @binding(1) var<storage, read_write> reduction: array<f32>;

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
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let anchor_id = global_id.x;
    let lid = local_id.x;
    let n = config.n;
    let window_size = config.window_size;
    let num_anchors = arrayLength(&anchor_indices);

    // Initial window bounds for this thread
    var left = 0u;
    var right = 0u;
    var x_i = 0.0;
    var valid_thread = false;

    if (anchor_id < num_anchors) {
        let i = anchor_indices[anchor_id];
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
        valid_thread = true;
    }

    // Determine workgroup bounds for tiling
    // We use thread 0 and thread 63 (or last valid thread)
    if (lid == 0u) {
        wg_min_left = left;
    }
    if (lid == 63u || anchor_id == num_anchors - 1u) {
        wg_max_right = right;
    }
    workgroupBarrier();

    // If whole workgroup is invalid, just exit
    if (anchor_id - lid >= num_anchors) {
        return;
    }

    // Calculate bandwidth from global memory before tiling (simplified)
    // We need bandwidth = max(abs(x_i - x[left]), abs(x_i - x[right]))
    var bandwidth = 0.0;
    if (valid_thread) {
        bandwidth = max(abs(x_i - x[left]), abs(x_i - x[right]));
        if (bandwidth <= 0.0) {
            anchor_output[anchor_id] = y[anchor_indices[anchor_id]];
            valid_thread = false; // Stop further regression calculation
        }
    }

    // Weighted linear regression variables
    var sum_w = 0.0;
    var sum_wx = 0.0;
    var sum_wxx = 0.0;
    var sum_wy = 0.0;
    var sum_wxy = 0.0;

    // Tiled processing from wg_min_left to wg_max_right
    let tile_start = (wg_min_left / 256u) * 256u;
    for (var t = tile_start; t <= wg_max_right; t += 256u) {
        // Load tile into shared memory
        for (var l = lid; l < 256u; l += 64u) {
            let idx = t + l;
            if (idx < n) {
                s_x[l] = x[idx];
                s_y[l] = y[idx];
                s_w[l] = robustness_weights[idx];
            } else {
                s_x[l] = 0.0;
                s_y[l] = 0.0;
                s_w[l] = 0.0;
            }
        }
        workgroupBarrier();

        // Process tile if thread is within its own window
        if (valid_thread) {
            let start_in_tile = max(t, left);
            let end_in_tile = min(t + 255u, right);
            
            for (var idx = start_in_tile; idx <= end_in_tile; idx++) {
                let l_idx = idx - t;
                let xj = s_x[l_idx];
                let yj = s_y[l_idx];
                let rw = s_w[l_idx];
                
                let dist = abs(xj - x_i);
                let u = dist / bandwidth;
                
                var w = 0.0;
                if (u < 1.0) {
                    let tmp = 1.0 - u * u * u;
                    w = tmp * tmp * tmp;
                }
                
                let combined_w = w * rw;
                
                sum_w += combined_w;
                sum_wx += combined_w * xj;
                sum_wxx += combined_w * xj * xj;
                sum_wy += combined_w * yj;
                sum_wxy += combined_w * xj * yj;
            }
        }
        workgroupBarrier();
    }

    // Finalize regression
    if (valid_thread) {
        if (sum_w <= 0.0) {
            anchor_output[anchor_id] = y[anchor_indices[anchor_id]];
        } else {
            let det = sum_w * sum_wxx - sum_wx * sum_wx;
            if (abs(det) < 1e-10) {
                anchor_output[anchor_id] = sum_wy / sum_w;
            } else {
                let a = (sum_wy * sum_wxx - sum_wxy * sum_wx) / det;
                let b = (sum_w * sum_wxy - sum_wx * sum_wy) / det;
                anchor_output[anchor_id] = a + b * x_i;
            }
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
// Kernel 3: MAR Reduction
// -----------------------------------------------------------------------------
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_sum_abs(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let i = global_id.x;
    var val = 0.0;
    if (i < config.n) {
        val = abs(residuals[i]);
    }
    
    scratch[local_id.x] = val;
    workgroupBarrier();
    
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (local_id.x < s) {
            scratch[local_id.x] += scratch[local_id.x + s];
        }
        workgroupBarrier();
    }
    
    if (local_id.x == 0u) {
        reduction[workgroup_id.x] = scratch[0];
    }
}

// -----------------------------------------------------------------------------
// Kernel 4: Finalize Scale
// -----------------------------------------------------------------------------
@compute @workgroup_size(1)
fn finalize_scale() {
    var total_sum = 0.0;
    let num_workgroups = (config.n + 255u) / 256u;
    for (var i = 0u; i < num_workgroups; i = i + 1u) {
        total_sum += reduction[i];
    }
    let mar = total_sum / f32(config.n);
    w_config.scale = max(mar, 1e-10);
}

// -----------------------------------------------------------------------------
// Kernel 5: Update Weights
// -----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn update_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= w_config.n) { return; }

    let r = abs(residuals[i]);
    let tuned_scale = w_config.scale * 6.0;

    if (tuned_scale <= 1e-12) {
        robustness_weights[i] = 1.0;
    } else {
        let u = r / tuned_scale;
        if (u < 1.0) {
            let tmp = 1.0 - u * u;
            robustness_weights[i] = tmp * tmp;
        } else {
            robustness_weights[i] = 0.0;
        }
    }
}
"#;

thread_local! {
    static THREAD_EXECUTOR: std::cell::RefCell<Option<GpuExecutor>> = const { std::cell::RefCell::new(None) };
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuConfig {
    n: u32,
    window_size: u32,
    weight_function: u32,
    zero_weight_fallback: u32,
    fraction: f32,
    delta: f32,
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WeightConfig {
    n: u32,
    scale: f32,
}

struct GpuExecutor {
    device: Device,
    queue: Queue,

    // Pipelines
    fit_pipeline: ComputePipeline,
    interpolate_pipeline: ComputePipeline,
    weight_pipeline: ComputePipeline,
    mar_pipeline: ComputePipeline,
    finalize_pipeline: ComputePipeline,

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
    residuals_buffer: Option<Buffer>,

    // Buffers - Group 3
    w_config_buffer: Option<Buffer>,
    reduction_buffer: Option<Buffer>,

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

        // Handle Option or Result depending on wgpu version?
        // Error message said: `Result<Adapter, RequestAdapterError>`
        // So we just use `?` or map_err.
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
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
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
            mar_pipeline: create_pipeline("reduce_sum_abs"),
            finalize_pipeline: create_pipeline("finalize_scale"),
            weight_pipeline: create_pipeline("update_weights"),
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
            residuals_buffer: None,
            w_config_buffer: None,
            reduction_buffer: None,
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
        if let Some(buffer) = buffer_opt.as_ref() {
            if buffer.size() < size_required {
                *buffer_opt = None;
            }
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

    fn reset_buffers(
        &mut self,
        x: &[f32],
        y: &[f32],
        anchors: &[u32],
        intervals: &[u32],
        rob_w: &[f32],
        config: GpuConfig,
    ) {
        let n = x.len() as u32;
        let num_anchors = anchors.len() as u32;
        let n_bytes = (n as usize * 4) as u64;
        let anchor_bytes = (num_anchors as usize * 4) as u64;

        let mut bg_needs_update = false;

        // Group 0: Config (Uniform)
        if Self::ensure_buffer_capacity(
            &self.device,
            "Config",
            &mut self.config_buffer,
            std::mem::size_of::<GpuConfig>() as u64,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }
        self.queue.write_buffer(
            self.config_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[config]),
        );

        // Group 0: X, Y, Anchors, AnchorOutput
        if Self::ensure_buffer_capacity(
            &self.device,
            "X",
            &mut self.x_buffer,
            n_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }
        self.queue
            .write_buffer(self.x_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(x));

        if Self::ensure_buffer_capacity(
            &self.device,
            "Y",
            &mut self.y_buffer,
            n_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }
        self.queue
            .write_buffer(self.y_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(y));

        if Self::ensure_buffer_capacity(
            &self.device,
            "AnchorIndices",
            &mut self.anchor_indices_buffer,
            anchor_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }
        self.queue.write_buffer(
            self.anchor_indices_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(anchors),
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
        if Self::ensure_buffer_capacity(
            &self.device,
            "IntervalMap",
            &mut self.interval_map_buffer,
            n_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }
        self.queue.write_buffer(
            self.interval_map_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(intervals),
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
            n_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        ) {
            bg_needs_update = true;
        }
        self.queue.write_buffer(
            self.weights_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(rob_w),
        );

        if Self::ensure_buffer_capacity(
            &self.device,
            "YSmooth",
            &mut self.y_smooth_buffer,
            n_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        if Self::ensure_buffer_capacity(
            &self.device,
            "Residuals",
            &mut self.residuals_buffer,
            n_bytes,
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
                ],
            }));
        }

        // Group 3: Aux
        bg_needs_update = false;
        if Self::ensure_buffer_capacity(
            &self.device,
            "WConfig",
            &mut self.w_config_buffer,
            std::mem::size_of::<WeightConfig>() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ) {
            bg_needs_update = true;
        }

        let reduction_size = (n.div_ceil(256) as usize * 4) as u64;
        if Self::ensure_buffer_capacity(
            &self.device,
            "Reduction",
            &mut self.reduction_buffer,
            reduction_size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
                ],
            }));
        }

        // Staging
        Self::ensure_buffer_capacity(
            &self.device,
            "Staging",
            &mut self.staging_buffer,
            n_bytes,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        self.n = n;
        self.num_anchors = num_anchors;
    }

    fn record_pipeline(
        &self,
        encoder: &mut wgpu::CommandEncoder,
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

    fn record_iteration(&self, encoder: &mut wgpu::CommandEncoder) {
        // 1. Fit at Anchors
        self.record_pipeline(encoder, &self.fit_pipeline, self.num_anchors.div_ceil(64));
        // 2. Interpolate
        self.record_pipeline(encoder, &self.interpolate_pipeline, self.n.div_ceil(64));
        // 3. Compute Scale (MAR)
        self.record_pipeline(encoder, &self.mar_pipeline, self.n.div_ceil(256));
        self.record_pipeline(encoder, &self.finalize_pipeline, 1);
        // 4. Update Weights
        self.record_pipeline(encoder, &self.weight_pipeline, self.n.div_ceil(64));
    }

    async fn download_buffer(&self, buf: &Buffer) -> Option<Vec<f32>> {
        let size = (self.n as usize * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, self.staging_buffer.as_ref().unwrap(), 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.as_ref().unwrap().slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(MapMode::Read, move |v| tx.send(v).unwrap());
        let _ = self.device.poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Some(Ok(())) = rx.receive().await {
            let data = slice.get_mapped_range();
            let ret = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.staging_buffer.as_ref().unwrap().unmap();
            Some(ret)
        } else {
            None
        }
    }
}

/// Perform a GPU-accelerated LOWESS fit pass.
pub fn fit_pass_gpu<T>(
    x: &[T],
    y: &[T],
    config: &LowessConfig<T>,
) -> (Vec<T>, Option<Vec<T>>, usize, Vec<T>)
where
    T: Float + Debug + Send + Sync + 'static,
{
    {
        use pollster::block_on;

        // Persistent Thread-Local Executor
        THREAD_EXECUTOR.with(|cell| {
            let mut opt = cell.borrow_mut();
            if opt.is_none() {
                *opt = block_on(GpuExecutor::new()).ok();
            }

            let exec = opt.as_mut().expect("Failed to initialize GPU executor");

            let x_f32: Vec<f32> = x.iter().map(|v| v.to_f32().unwrap()).collect();
            let y_f32: Vec<f32> = y.iter().map(|v| v.to_f32().unwrap()).collect();
            let n = x.len();
            let delta = config.delta.to_f32().unwrap();

            // Compute Anchors & Intervals
            let mut anchors = Vec::with_capacity(n / 10);
            let mut intervals = vec![0u32; n];

            let mut last_idx = 0;
            anchors.push(0);
            let mut current_anchor_idx = 0;

            for i in 0..n {
                if i > 0 && x_f32[i] - x_f32[last_idx] > delta {
                    last_idx = i;
                    anchors.push(i as u32);
                    current_anchor_idx += 1;
                }
                if current_anchor_idx >= 1 {
                    intervals[i] = current_anchor_idx - 1;
                } else {
                    intervals[i] = 0;
                }
            }
            // Ensure last point is anchor if not already
            if *anchors.last().unwrap() != (n - 1) as u32 {
                anchors.push((n - 1) as u32);
                // Fix intervals for tail? Logic above is simple "closest left anchor".
                // Correct interval logic for interpolation between A[k] and A[k+1]:
                // Point i must have intervals[i] = k where A[k] <= i <= A[k+1].
                // Re-run interval mapping strictly.
            }

            // Strict Interval Mapping
            let mut anchor_ptr = 0;
            for (i, interval) in intervals.iter_mut().enumerate() {
                while anchor_ptr + 1 < anchors.len() && (i as u32) >= anchors[anchor_ptr + 1] {
                    anchor_ptr += 1;
                }
                // For i between A[ptr] and A[ptr+1], interval is ptr.
                *interval = anchor_ptr as u32;
            }

            let gpu_config = GpuConfig {
                n: n as u32,
                window_size: (config.fraction.unwrap().to_f32().unwrap() * n as f32) as u32,
                weight_function: 0,
                zero_weight_fallback: 0,
                fraction: config.fraction.unwrap().to_f32().unwrap(),
                delta,
                padding: [0, 0],
            };

            exec.reset_buffers(
                &x_f32,
                &y_f32,
                &anchors,
                &intervals,
                &vec![1.0; n],
                gpu_config,
            );

            let mut encoder = exec
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("LOWESS Main"),
                });

            for _ in 0..=config.iterations {
                exec.record_iteration(&mut encoder);
            }
            exec.queue.submit(Some(encoder.finish()));

            let y_res =
                block_on(exec.download_buffer(exec.y_smooth_buffer.as_ref().unwrap())).unwrap();
            let w_res =
                block_on(exec.download_buffer(exec.weights_buffer.as_ref().unwrap())).unwrap();

            let y_out: Vec<T> = y_res.into_iter().map(|v| T::from(v).unwrap()).collect();
            let w_out: Vec<T> = w_res.into_iter().map(|v| T::from(v).unwrap()).collect();

            (y_out, None, config.iterations, w_out)
        })
    }
    #[cfg(not(feature = "gpu"))]
    {
        unimplemented!("GPU feature disabled")
    }
}
