"""
	FastLOWESS

High-performance LOWESS (Locally Weighted Scatterplot Smoothing) for Julia.

Provides bindings to the fastlowess Rust library for fast, robust LOWESS smoothing.

# Main API
- `Lowess(; kwargs...)`: Configure batch LOWESS
- `fit(model, x, y)`: Fit and return results
- `StreamingLowess(; kwargs...)` / `process_chunk` / `finalize`: Streaming mode
- `OnlineLowess(; kwargs...)` / `add_point`: Online sliding-window mode

# Example
```julia
using FastLOWESS

x = collect(1.0:0.1:10.0)
y = sin.(x) .+ 0.1 .* randn(length(x))

result = fit(Lowess(fraction=0.3), x, y)
println("Smoothed values: ", result.y)
```
"""
module FastLOWESS

export Lowess, StreamingLowess, OnlineLowess
export fit, process_chunk, finalize, add_point
export LowessResult, OnlineOutput, Diagnostics

import Base: finalize

# Try to import JLL package first
try
	using fastlowess_jll
catch e
	# JLL not available, will use fallback
end

# Library name varies by platform
const LIBNAME =
	Sys.iswindows() ? "fastlowess_jl.dll" :
	Sys.isapple() ? "libfastlowess_jl.dylib" : "libfastlowess_jl.so"

# Try to load from JLL package first, fall back to local build
function find_library()
	# Option 1: Check environment variable (PRIORITY)
	if haskey(ENV, "FASTLOWESS_LIB")
		lib = ENV["FASTLOWESS_LIB"]
		@info "Using library from FASTLOWESS_LIB: $lib"
		return lib
	end

	# Option 2: Use JLL package if available (for registered package)
	if @isdefined(fastlowess_jll)
		try
			if hasproperty(fastlowess_jll, :libfastlowess_jl)
				lib = fastlowess_jll.libfastlowess_jl
				@info "Using fastlowess_jll library: $lib"
				return lib
			end
		catch e
			@warn "Failed to load from fastlowess_jll" exception = e
		end
	end

	# Option 3: Check relative paths (development mode)
	# Path: julia/src/fastlowess.jl -> julia/ -> bindings/julia/ -> bindings/ -> lowess-project/
	src_dir = @__DIR__                        # julia/src/
	julia_dir = dirname(src_dir)              # julia/
	bindings_julia_dir = dirname(julia_dir)   # bindings/julia/
	bindings_dir = dirname(bindings_julia_dir)# bindings/
	workspace_root = dirname(bindings_dir)    # lowess-project/

	candidates = [
		# Workspace root target (most common for workspace members)
		joinpath(workspace_root, "target", "release", LIBNAME),
		joinpath(workspace_root, "target", "debug", LIBNAME),
		# Local target (if built standalone)
		joinpath(bindings_julia_dir, "target", "release", LIBNAME),
		joinpath(bindings_julia_dir, "target", "debug", LIBNAME),
		# Same directory as module
		joinpath(julia_dir, LIBNAME),
	]

	for path ∈ candidates
		if isfile(path)
			@info "Using local library: $path"
			return path
		end
	end

	# Fall back to system path
	@warn "Library not found in JLL or local paths, falling back to system path"
	return LIBNAME
end

const libfastlowess = Ref("")

function current_library()
	if isempty(libfastlowess[])
		libfastlowess[] = find_library()
	end
	return libfastlowess[]
end

function __init__()
	libfastlowess[] = find_library()
end

"""
	Diagnostics

Diagnostic statistics for LOWESS fit quality.

# Fields
- `rmse::Float64`: Root Mean Squared Error
- `mae::Float64`: Mean Absolute Error
- `r_squared::Float64`: R-squared (coefficient of determination)
- `aic::Float64`: Akaike Information Criterion (NaN if not computed)
- `aicc::Float64`: Corrected AIC (NaN if not computed)
- `effective_df::Float64`: Effective degrees of freedom (NaN if not computed)
- `residual_sd::Float64`: Residual standard deviation
"""
struct Diagnostics
	rmse::Float64
	mae::Float64
	r_squared::Float64
	aic::Float64
	aicc::Float64
	effective_df::Float64
	residual_sd::Float64
end

"""
	LowessResult

Result from LOWESS smoothing.

# Fields
- `x::Vector{Float64}`: Sorted x values
- `y::Vector{Float64}`: Smoothed y values
- `standard_errors::Union{Vector{Float64}, Nothing}`: Standard errors (if computed)
- `confidence_lower::Union{Vector{Float64}, Nothing}`: Lower confidence bounds
- `confidence_upper::Union{Vector{Float64}, Nothing}`: Upper confidence bounds
- `prediction_lower::Union{Vector{Float64}, Nothing}`: Lower prediction bounds
- `prediction_upper::Union{Vector{Float64}, Nothing}`: Upper prediction bounds
- `residuals::Union{Vector{Float64}, Nothing}`: Residuals
- `robustness_weights::Union{Vector{Float64}, Nothing}`: Robustness weights
- `cv_scores::Union{Vector{Float64}, Nothing}`: Cross-validation scores for tested fractions
- `fraction_used::Float64`: Fraction used for smoothing
- `iterations_used::Int`: Number of iterations performed (-1 if not available)
- `diagnostics::Union{Diagnostics, Nothing}`: Diagnostic metrics
"""
struct LowessResult
	x::Vector{Float64}
	y::Vector{Float64}
	standard_errors::Union{Vector{Float64}, Nothing}
	confidence_lower::Union{Vector{Float64}, Nothing}
	confidence_upper::Union{Vector{Float64}, Nothing}
	prediction_lower::Union{Vector{Float64}, Nothing}
	prediction_upper::Union{Vector{Float64}, Nothing}
	residuals::Union{Vector{Float64}, Nothing}
	robustness_weights::Union{Vector{Float64}, Nothing}
	cv_scores::Union{Vector{Float64}, Nothing}
	fraction_used::Float64
	iterations_used::Int
	diagnostics::Union{Diagnostics, Nothing}
end

"""
	OnlineOutput

Result from a single `add_point` call.

# Fields
- `smoothed::Float64`: Smoothed value for the latest point
- `std_error::Union{Float64, Nothing}`: Standard error (if computed)
- `residual::Union{Float64, Nothing}`: Residual y − smoothed (if computed)
- `robustness_weight::Union{Float64, Nothing}`: Robustness weight (if computed)
- `iterations_used::Union{Int, Nothing}`: Number of robustness iterations
"""
struct OnlineOutput
	smoothed::Float64
	std_error::Union{Float64, Nothing}
	residual::Union{Float64, Nothing}
	robustness_weight::Union{Float64, Nothing}
	iterations_used::Union{Int, Nothing}
end

# C FFI struct for per-point online output (must match Rust definition).
struct CJlOnlineOutput
	has_value::Cint
	smoothed::Cdouble
	std_error::Cdouble
	residual::Cdouble
	robustness_weight::Cdouble
	iterations_used::Cint
	error::Ptr{Cchar}
end

# C FFI result struct (must match Rust definition)
struct CJlLowessResult
	x::Ptr{Cdouble}
	y::Ptr{Cdouble}
	n::Culong
	standard_errors::Ptr{Cdouble}
	confidence_lower::Ptr{Cdouble}
	confidence_upper::Ptr{Cdouble}
	prediction_lower::Ptr{Cdouble}
	prediction_upper::Ptr{Cdouble}
	residuals::Ptr{Cdouble}
	robustness_weights::Ptr{Cdouble}
	cv_scores::Ptr{Cdouble}
	cv_scores_len::Culong
	fraction_used::Cdouble
	iterations_used::Cint
	rmse::Cdouble
	mae::Cdouble
	r_squared::Cdouble
	aic::Cdouble
	aicc::Cdouble
	effective_df::Cdouble
	residual_sd::Cdouble
	error::Ptr{Cchar}
end

function ptr_to_vector(ptr::Ptr{Cdouble}, n::Int)
	if ptr == C_NULL
		return nothing
	end
	return unsafe_wrap(Array, ptr, n, own = false) |> copy
end

function convert_result(c_result::CJlLowessResult)
	# Check for error
	if c_result.error != Ptr{Cchar}(C_NULL)
		error_msg = unsafe_string(Ptr{UInt8}(c_result.error))
		# Free the result before throwing
		@ccall current_library().jl_lowess_free_result(
			Ref(c_result)::Ptr{CJlLowessResult},
		)::Cvoid
		error("fastlowess error: $error_msg")
	end

	n = Int(c_result.n)

	# Extract arrays
	x = ptr_to_vector(c_result.x, n)
	y = ptr_to_vector(c_result.y, n)

	if x === nothing || y === nothing
		@ccall current_library().jl_lowess_free_result(
			Ref(c_result)::Ptr{CJlLowessResult},
		)::Cvoid
		error("fastlowess error: result arrays are null")
	end

	x = x::Vector{Float64}
	y = y::Vector{Float64}

	standard_errors = ptr_to_vector(c_result.standard_errors, n)
	confidence_lower = ptr_to_vector(c_result.confidence_lower, n)
	confidence_upper = ptr_to_vector(c_result.confidence_upper, n)
	prediction_lower = ptr_to_vector(c_result.prediction_lower, n)
	prediction_upper = ptr_to_vector(c_result.prediction_upper, n)
	residuals = ptr_to_vector(c_result.residuals, n)
	robustness_weights = ptr_to_vector(c_result.robustness_weights, n)
	cv_scores = ptr_to_vector(c_result.cv_scores, Int(c_result.cv_scores_len))

	# Extract diagnostics
	diagnostics = if !isnan(c_result.rmse)
		Diagnostics(
			c_result.rmse,
			c_result.mae,
			c_result.r_squared,
			c_result.aic,
			c_result.aicc,
			c_result.effective_df,
			c_result.residual_sd,
		)
	else
		nothing
	end

	result = LowessResult(
		x,
		y,
		standard_errors,
		confidence_lower,
		confidence_upper,
		prediction_lower,
		prediction_upper,
		residuals,
		robustness_weights,
		cv_scores,
		c_result.fraction_used,
		Int(c_result.iterations_used),
		diagnostics,
	)

	# Free the C result
	@ccall current_library().jl_lowess_free_result(
		Ref(c_result)::Ptr{CJlLowessResult},
	)::Cvoid

	return result
end

"""
	Base.append!(a::LowessResult, b::LowessResult) -> LowessResult

Append the results from `b` to `a`. This modifies `a` in place.
"""
function Base.append!(a::LowessResult, b::LowessResult)
	append!(a.x, b.x)
	append!(a.y, b.y)

	if a.standard_errors !== nothing && b.standard_errors !== nothing
		append!(a.standard_errors, b.standard_errors)
	end
	if a.confidence_lower !== nothing && b.confidence_lower !== nothing
		append!(a.confidence_lower, b.confidence_lower)
	end
	if a.confidence_upper !== nothing && b.confidence_upper !== nothing
		append!(a.confidence_upper, b.confidence_upper)
	end
	if a.prediction_lower !== nothing && b.prediction_lower !== nothing
		append!(a.prediction_lower, b.prediction_lower)
	end
	if a.prediction_upper !== nothing && b.prediction_upper !== nothing
		append!(a.prediction_upper, b.prediction_upper)
	end
	if a.residuals !== nothing && b.residuals !== nothing
		append!(a.residuals, b.residuals)
	end
	if a.robustness_weights !== nothing && b.robustness_weights !== nothing
		append!(a.robustness_weights, b.robustness_weights)
	end

	# Update fraction_used and iterations_used if they differ?
	# Streaming usually keeps them constant. We'll keep a's values.

	return a
end



"""
	Lowess(; kwargs...)

Stateful batch LOWESS smoother.

# Keyword Arguments
- `fraction::Float64 = 0.67`: Smoothing fraction (proportion of data used for each fit)
- `iterations::Int = 3`: Number of robustness iterations
- `delta::Float64 = NaN`: Interpolation optimization threshold (NaN for auto)
- `weight_function::String = "tricube"`: Kernel function
- `robustness_method::String = "bisquare"`: Robustness method
- `scaling_method::String = "mad"`: Scaling method for robustness
- `boundary_policy::String = "extend"`: Handling of edge effects
- `confidence_intervals::Float64 = NaN`: Confidence level (e.g., 0.95), NaN to disable
- `prediction_intervals::Float64 = NaN`: Prediction interval level, NaN to disable
- `return_diagnostics::Bool = false`: Whether to compute RMSE, MAE, R², etc.
- `return_residuals::Bool = false`: Whether to include residuals
- `return_robustness_weights::Bool = false`: Whether to include robustness weights
- `zero_weight_fallback::String = "use_local_mean"`: Fallback when all weights are zero
- `auto_converge::Float64 = NaN`: Tolerance for auto-convergence, NaN to disable
- `cv_fractions::Vector{Float64} = Float64[]`: Fractions for cross-validation
- `cv_method::String = "kfold"`: CV method ("loocv" or "kfold")
- `cv_k::Int = 5`: Number of folds for k-fold CV
- `parallel::Bool = true`: Enable parallel execution
- `cv_seed::Union{Int,Nothing} = nothing`: Seed for reproducible K-fold splits
- `return_se::Bool = false`: Return standard errors

# Example
```julia
l = Lowess(fraction=0.3)
result = fit(l, x, y)
```
"""
mutable struct Lowess
	handle::Ptr{Cvoid}

	function Lowess(;
		fraction::Float64 = 0.67,
		iterations::Int = 3,
		delta::Float64 = NaN,
		weight_function::String = "tricube",
		robustness_method::String = "bisquare",
		scaling_method::String = "mad",
		boundary_policy::String = "extend",
		confidence_intervals::Float64 = NaN,
		prediction_intervals::Float64 = NaN,
		return_diagnostics::Bool = false,
		return_residuals::Bool = false,
		return_robustness_weights::Bool = false,
		zero_weight_fallback::String = "use_local_mean",
		auto_converge::Float64 = NaN,
		cv_fractions::Vector{Float64} = Float64[],
		cv_method::String = "kfold",
		cv_k::Int = 5,
		parallel::Bool = true,
		cv_seed::Union{Int, Nothing} = nothing,
		return_se::Bool = false,
	)
		cv_ptr = isempty(cv_fractions) ? Ptr{Cdouble}(C_NULL) : pointer(cv_fractions)
		cv_len = length(cv_fractions)

		handle = @ccall current_library().jl_lowess_new(
			fraction::Cdouble,
			Cint(iterations)::Cint,
			delta::Cdouble,
			weight_function::Cstring,
			robustness_method::Cstring,
			scaling_method::Cstring,
			boundary_policy::Cstring,
			confidence_intervals::Cdouble,
			prediction_intervals::Cdouble,
			Cint(return_diagnostics)::Cint,
			Cint(return_residuals)::Cint,
			Cint(return_robustness_weights)::Cint,
			zero_weight_fallback::Cstring,
			auto_converge::Cdouble,
			cv_ptr::Ptr{Cdouble},
			Culong(cv_len)::Culong,
			cv_method::Cstring,
			Cint(cv_k)::Cint,
			Cint(parallel)::Cint,
			Culong(cv_seed !== nothing ? cv_seed : 0)::Culong,
			Cint(return_se)::Cint,
		)::Ptr{Cvoid}

		if handle == C_NULL
			error("Failed to create Lowess configuration")
		end

		obj = new(handle)
		finalizer(
			x -> @ccall(current_library().jl_lowess_free(x.handle::Ptr{Cvoid})::Cvoid),
			obj,
		)
		return obj
	end
end

"""
	fit(l::Lowess, x, y; custom_weights=nothing) -> LowessResult

Fit the LOWESS model to data.

# Arguments
- `custom_weights::Union{Vector{Float64}, Nothing} = nothing`: Per-observation
  case weights. Must have the same length as `x` and `y`. All values must be
  finite and non-negative. Pass `nothing` (default) to disable.
"""
function fit(
	l::Lowess,
	x::Vector{Float64},
	y::Vector{Float64};
	custom_weights::Union{Vector{Float64}, Nothing} = nothing,
)
	n = length(x)
	if n != length(y)
		throw(ArgumentError("x and y must have the same length"))
	end

	if custom_weights !== nothing
		if length(custom_weights) != n
			throw(ArgumentError("custom_weights must have the same length as y"))
		end
	end

	c_result = @ccall current_library().jl_lowess_fit(
		l.handle::Ptr{Cvoid},
		x::Ptr{Cdouble},
		y::Ptr{Cdouble},
		Culong(n)::Culong,
		(
			custom_weights !== nothing ? pointer(custom_weights) : Ptr{Cdouble}(C_NULL)
		)::Ptr{Cdouble},
		Culong(custom_weights !== nothing ? length(custom_weights) : 0)::Culong,
	)::CJlLowessResult

	return convert_result(c_result)
end

"""
	StreamingLowess(; kwargs...)

Stateful streaming LOWESS smoother.

# Keyword Arguments
- `fraction::Float64 = 0.67`: Smoothing fraction
- `chunk_size::Int = 5000`: Size of each processing chunk
- `overlap::Int = 500`: Overlap between adjacent chunks
- `iterations::Int = 3`: Number of robustness iterations
- `delta::Float64 = NaN`: Interpolation threshold (NaN for auto)
- `weight_function::String = "tricube"`: Kernel function
- `robustness_method::String = "bisquare"`: Robustness method
- `scaling_method::String = "mad"`: Scaling method
- `boundary_policy::String = "extend"`: Boundary handling
- `auto_converge::Float64 = NaN`: Auto-convergence tolerance
- `return_diagnostics::Bool = false`: Compute diagnostics
- `return_residuals::Bool = false`: Include residuals
- `return_robustness_weights::Bool = false`: Include weights
- `zero_weight_fallback::String = "use_local_mean"`: Zero weight handling
- `merge_strategy::String = "weighted_average"`: Strategy for merging overlapping chunk
  regions: "average", "weighted_average", "take_first", "take_last"
- `parallel::Bool = true`: Enable parallel execution
"""
mutable struct StreamingLowess
	handle::Ptr{Cvoid}

	function StreamingLowess(;
		fraction::Float64 = 0.67,
		chunk_size::Int = 5000,
		overlap::Int = 500,
		iterations::Int = 3,
		delta::Float64 = NaN,
		weight_function::String = "tricube",
		robustness_method::String = "bisquare",
		scaling_method::String = "mad",
		boundary_policy::String = "extend",
		auto_converge::Float64 = NaN,
		return_diagnostics::Bool = false,
		return_residuals::Bool = false,
		return_robustness_weights::Bool = false,
		zero_weight_fallback::String = "use_local_mean",
		merge_strategy::String = "weighted_average",
		parallel::Bool = true,
	)
		handle = @ccall current_library().jl_streaming_lowess_new(
			fraction::Cdouble,
			Cint(chunk_size)::Cint,
			Cint(overlap)::Cint,
			Cint(iterations)::Cint,
			delta::Cdouble,
			weight_function::Cstring,
			robustness_method::Cstring,
			scaling_method::Cstring,
			boundary_policy::Cstring,
			auto_converge::Cdouble,
			Cint(return_diagnostics)::Cint,
			Cint(return_residuals)::Cint,
			Cint(return_robustness_weights)::Cint,
			zero_weight_fallback::Cstring,
			merge_strategy::Cstring,
			Cint(parallel)::Cint,
		)::Ptr{Cvoid}

		if handle == C_NULL
			error("Failed to create StreamingLowess")
		end

		obj = new(handle)
		finalizer(
			x -> @ccall(
				current_library().jl_streaming_lowess_free(x.handle::Ptr{Cvoid})::Cvoid
			),
			obj,
		)
		return obj
	end
end

"""
	process_chunk(s::StreamingLowess, x, y) -> LowessResult

Process a chunk of data.
"""
function process_chunk(s::StreamingLowess, x::Vector{Float64}, y::Vector{Float64})
	n = length(x)
	if n != length(y)
		throw(ArgumentError("x and y must have the same length"))
	end

	c_result = @ccall current_library().jl_streaming_lowess_process_chunk(
		s.handle::Ptr{Cvoid},
		x::Ptr{Cdouble},
		y::Ptr{Cdouble},
		Culong(n)::Culong,
	)::CJlLowessResult

	return convert_result(c_result)
end

"""
	finalize(s::StreamingLowess) -> LowessResult

Finalize streaming and return remaining buffered data.
"""
function finalize(s::StreamingLowess)
	c_result = @ccall current_library().jl_streaming_lowess_finalize(
		s.handle::Ptr{Cvoid},
	)::CJlLowessResult

	return convert_result(c_result)
end

"""
	OnlineLowess(; kwargs...)

Stateful online LOWESS smoother.

# Keyword Arguments
- `fraction::Float64 = 0.67`: Smoothing fraction
- `window_capacity::Int = 1000`: Maximum points to retain in window
- `min_points::Int = 3`: Minimum points before smoothing starts
- `iterations::Int = 3`: Number of robustness iterations
- `delta::Float64 = NaN`: Interpolation threshold (NaN for auto)
- `weight_function::String = "tricube"`: Kernel function
- `robustness_method::String = "bisquare"`: Robustness method
- `scaling_method::String = "mad"`: Scaling method
- `boundary_policy::String = "extend"`: Boundary handling
- `update_mode::String = "full"`: Update strategy ("full" or "incremental")
- `auto_converge::Float64 = NaN`: Auto-convergence tolerance
- `return_robustness_weights::Bool = false`: Include weights
- `return_diagnostics::Bool = false`: Compute diagnostics (RMSE, MAE, R², etc.)
- `return_residuals::Bool = false`: Include residuals in the result
- `zero_weight_fallback::String = "use_local_mean"`: Zero weight handling
- `parallel::Bool = false`: Enable parallel execution (default `false` for online mode;
  online LOWESS processes one point at a time and rarely benefits from parallelism)
"""
mutable struct OnlineLowess
	handle::Ptr{Cvoid}

	function OnlineLowess(;
		fraction::Float64 = 0.67,
		window_capacity::Int = 1000,
		min_points::Int = 3,
		iterations::Int = 3,
		delta::Float64 = NaN,
		weight_function::String = "tricube",
		robustness_method::String = "bisquare",
		scaling_method::String = "mad",
		boundary_policy::String = "extend",
		update_mode::String = "full",
		auto_converge::Float64 = NaN,
		return_robustness_weights::Bool = false,
		return_diagnostics::Bool = false,
		return_residuals::Bool = false,
		zero_weight_fallback::String = "use_local_mean",
		parallel::Bool = false,
	)
		handle = @ccall current_library().jl_online_lowess_new(
			fraction::Cdouble,
			Cint(window_capacity)::Cint,
			Cint(min_points)::Cint,
			Cint(iterations)::Cint,
			delta::Cdouble,
			weight_function::Cstring,
			robustness_method::Cstring,
			scaling_method::Cstring,
			boundary_policy::Cstring,
			update_mode::Cstring,
			auto_converge::Cdouble,
			Cint(return_robustness_weights)::Cint,
			Cint(return_diagnostics)::Cint,
			Cint(return_residuals)::Cint,
			zero_weight_fallback::Cstring,
			Cint(parallel)::Cint,
		)::Ptr{Cvoid}

		if handle == C_NULL
			error("Failed to create OnlineLowess")
		end

		obj = new(handle)
		finalizer(
			x -> @ccall(
				current_library().jl_online_lowess_free(x.handle::Ptr{Cvoid})::Cvoid
			),
			obj,
		)
		return obj
	end
end

"""
	add_point(o::OnlineLowess, x, y) -> Union{OnlineOutput, Nothing}

Add a single point to the online processor and return its smoothed value.
Returns `nothing` while the window is still filling (fewer than `min_points`
have been seen), and an `OnlineOutput` once smoothing begins.
"""
function add_point(o::OnlineLowess, x::Float64, y::Float64)
	c_result = @ccall current_library().jl_online_lowess_add_point(
		o.handle::Ptr{Cvoid},
		x::Cdouble,
		y::Cdouble,
	)::CJlOnlineOutput

	if c_result.error != Ptr{Cchar}(C_NULL)
		error_msg = unsafe_string(Ptr{UInt8}(c_result.error))
		@ccall current_library().jl_online_free_output(
			Ref(c_result)::Ptr{CJlOnlineOutput},
		)::Cvoid
		error("fastlowess error: $error_msg")
	end

	if c_result.has_value == 0
		return nothing
	end

	return OnlineOutput(
		c_result.smoothed,
		isnan(c_result.std_error) ? nothing : c_result.std_error,
		isnan(c_result.residual) ? nothing : c_result.residual,
		isnan(c_result.robustness_weight) ? nothing : c_result.robustness_weight,
		c_result.iterations_used == -1 ? nothing : Int(c_result.iterations_used),
	)
end

end # module FastLOWESS
