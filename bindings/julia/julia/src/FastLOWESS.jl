"""
	FastLOWESS

High-performance LOWESS (Locally Weighted Scatterplot Smoothing) for Julia.

Provides bindings to the fastlowess Rust library for fast, robust LOWESS smoothing.

# Main Functions
- `smooth(x, y; kwargs...)`: Batch LOWESS smoothing
- `smooth_streaming(x, y; kwargs...)`: Streaming mode for large datasets
- `smooth_online(x, y; kwargs...)`: Online mode with sliding window

# Example
```julia
using FastLOWESS

x = collect(1.0:0.1:10.0)
y = sin.(x) .+ 0.1 .* randn(length(x))

result = smooth(x, y, fraction=0.3)
println("Smoothed values: ", result.y)
```
"""
module FastLOWESS

export Lowess, StreamingLowess, OnlineLowess
export fit, process_chunk, finalize, add_point
export LowessResult, Diagnostics

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
	if c_result.error != C_NULL
		error_msg = unsafe_string(c_result.error)
		# Free the result before throwing
		ccall(
			(:jl_lowess_free_result, current_library()),
			Cvoid,
			(Ptr{CJlLowessResult},),
			Ref(c_result),
		)
		error("fastlowess error: $error_msg")
	end

	n = Int(c_result.n)

	# Extract arrays
	x = ptr_to_vector(c_result.x, n)
	y = ptr_to_vector(c_result.y, n)

	if x === nothing || y === nothing
		ccall(
			(:jl_lowess_free_result, current_library()),
			Cvoid,
			(Ptr{CJlLowessResult},),
			Ref(c_result),
		)
		error("fastlowess error: result arrays are null")
	end

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
	ccall(
		(:jl_lowess_free_result, current_library()),
		Cvoid,
		(Ptr{CJlLowessResult},),
		Ref(c_result),
	)

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
	)
		cv_ptr = isempty(cv_fractions) ? C_NULL : pointer(cv_fractions)
		cv_len = length(cv_fractions)

		handle = ccall(
			(:jl_lowess_new, current_library()),
			Ptr{Cvoid},
			(
				Cdouble,
				Cint,
				Cdouble,
				Cstring,
				Cstring,
				Cstring,
				Cstring,
				Cdouble,
				Cdouble,
				Cint,
				Cint,
				Cint,
				Cstring,
				Cdouble,
				Ptr{Cdouble},
				Culong,
				Cstring,
				Cint,
				Cint,
			),
			fraction,
			Cint(iterations),
			delta,
			weight_function,
			robustness_method,
			scaling_method,
			boundary_policy,
			confidence_intervals,
			prediction_intervals,
			Cint(return_diagnostics),
			Cint(return_residuals),
			Cint(return_robustness_weights),
			zero_weight_fallback,
			auto_converge,
			cv_ptr,
			Culong(cv_len),
			cv_method,
			Cint(cv_k),
			Cint(parallel),
		)

		if handle == C_NULL
			error("Failed to create Lowess configuration")
		end

		obj = new(handle)
		finalizer(
			x ->
				ccall((:jl_lowess_free, current_library()), Cvoid, (Ptr{Cvoid},), x.handle),
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
function fit(l::Lowess, x::Vector{Float64}, y::Vector{Float64};
		custom_weights::Union{Vector{Float64}, Nothing} = nothing)
	n = length(x)
	if n != length(y)
		throw(ArgumentError("x and y must have the same length"))
	end

	cw_ptr = isnothing(custom_weights) ? C_NULL : pointer(custom_weights)
	cw_len = isnothing(custom_weights) ? Culong(0) : Culong(length(custom_weights))

	c_result = ccall(
		(:jl_lowess_fit, current_library()),
		CJlLowessResult,
		(Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Culong, Ptr{Cdouble}, Culong),
		l.handle,
		x,
		y,
		Culong(n),
		cw_ptr,
		cw_len,
	)

	return convert_result(c_result)
end

"""
	StreamingLowess(; kwargs...)

Stateful streaming LOWESS smoother.

# Keyword Arguments
- `fraction::Float64 = 0.3`: Smoothing fraction
- `chunk_size::Int = 5000`: Size of each processing chunk
- `overlap::Int = -1`: Overlap between chunks (-1 for auto = 10% of chunk_size)
- `iterations::Int = 3`: Number of robustness iterations
- `delta::Float64 = NaN`: Interpolation threshold
- `weight_function::String = "tricube"`: Kernel function
- `robustness_method::String = "bisquare"`: Robustness method
- `scaling_method::String = "mad"`: Scaling method
- `boundary_policy::String = "extend"`: Boundary handling
- `auto_converge::Float64 = NaN`: Auto-convergence tolerance
- `return_diagnostics::Bool = false`: Compute diagnostics
- `return_residuals::Bool = false`: Include residuals
- `return_robustness_weights::Bool = false`: Include weights
- `zero_weight_fallback::String = "use_local_mean"`: Zero weight handling
- `parallel::Bool = true`: Enable parallel execution
"""
mutable struct StreamingLowess
	handle::Ptr{Cvoid}

	function StreamingLowess(;
		fraction::Float64 = 0.3,
		chunk_size::Int = 5000,
		overlap::Int = -1,
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
		parallel::Bool = true,
	)
		handle = ccall(
			(:jl_streaming_lowess_new, current_library()),
			Ptr{Cvoid},
			(
				Cdouble,
				Cint,
				Cint,
				Cint,
				Cdouble,
				Cstring,
				Cstring,
				Cstring,
				Cstring,
				Cdouble,
				Cint,
				Cint,
				Cint,
				Cstring,
				Cint,
			),
			fraction,
			Cint(chunk_size),
			Cint(overlap),
			Cint(iterations),
			delta,
			weight_function,
			robustness_method,
			scaling_method,
			boundary_policy,
			auto_converge,
			Cint(return_diagnostics),
			Cint(return_residuals),
			Cint(return_robustness_weights),
			zero_weight_fallback,
			Cint(parallel),
		)

		if handle == C_NULL
			error("Failed to create StreamingLowess")
		end

		obj = new(handle)
		finalizer(
			x -> ccall(
				(:jl_streaming_lowess_free, current_library()),
				Cvoid,
				(Ptr{Cvoid},),
				x.handle,
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

	c_result = ccall(
		(:jl_streaming_lowess_process_chunk, current_library()),
		CJlLowessResult,
		(Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Culong),
		s.handle,
		x,
		y,
		Culong(n),
	)

	return convert_result(c_result)
end

"""
	finalize(s::StreamingLowess) -> LowessResult

Finalize streaming and return remaining buffered data.
"""
function finalize(s::StreamingLowess)
	c_result = ccall(
		(:jl_streaming_lowess_finalize, current_library()),
		CJlLowessResult,
		(Ptr{Cvoid},),
		s.handle,
	)

	return convert_result(c_result)
end

"""
	OnlineLowess(; kwargs...)

Stateful online LOWESS smoother.

# Keyword Arguments
- `fraction::Float64 = 0.2`: Smoothing fraction
- `window_capacity::Int = 100`: Maximum points to retain in window
- `min_points::Int = 2`: Minimum points before smoothing starts
- `iterations::Int = 3`: Number of robustness iterations
- `delta::Float64 = NaN`: Interpolation threshold
- `weight_function::String = "tricube"`: Kernel function
- `robustness_method::String = "bisquare"`: Robustness method
- `scaling_method::String = "mad"`: Scaling method
- `boundary_policy::String = "extend"`: Boundary handling
- `update_mode::String = "full"`: Update strategy ("full" or "incremental")
- `auto_converge::Float64 = NaN`: Auto-convergence tolerance
- `return_robustness_weights::Bool = false`: Include weights
- `zero_weight_fallback::String = "use_local_mean"`: Zero weight handling
- `parallel::Bool = false`: Enable parallel execution
"""
mutable struct OnlineLowess
	handle::Ptr{Cvoid}

	function OnlineLowess(;
		fraction::Float64 = 0.2,
		window_capacity::Int = 100,
		min_points::Int = 2,
		iterations::Int = 3,
		delta::Float64 = NaN,
		weight_function::String = "tricube",
		robustness_method::String = "bisquare",
		scaling_method::String = "mad",
		boundary_policy::String = "extend",
		update_mode::String = "full",
		auto_converge::Float64 = NaN,
		return_robustness_weights::Bool = false,
		zero_weight_fallback::String = "use_local_mean",
		parallel::Bool = false,
	)
		handle = ccall(
			(:jl_online_lowess_new, current_library()),
			Ptr{Cvoid},
			(
				Cdouble,
				Cint,
				Cint,
				Cint,
				Cdouble,
				Cstring,
				Cstring,
				Cstring,
				Cstring,
				Cstring,
				Cdouble,
				Cint,
				Cstring,
				Cint,
			),
			fraction,
			Cint(window_capacity),
			Cint(min_points),
			Cint(iterations),
			delta,
			weight_function,
			robustness_method,
			scaling_method,
			boundary_policy,
			update_mode,
			auto_converge,
			Cint(return_robustness_weights),
			zero_weight_fallback,
			Cint(parallel),
		)

		if handle == C_NULL
			error("Failed to create OnlineLowess")
		end

		obj = new(handle)
		finalizer(
			x -> ccall(
				(:jl_online_lowess_free, current_library()),
				Cvoid,
				(Ptr{Cvoid},),
				x.handle,
			),
			obj,
		)
		return obj
	end
end

"""
	add_point(o::OnlineLowess, x, y) -> Union{Float64, Nothing}

Add a single point to the online processor. Returns the smoothed value,
or `nothing` if not enough points are available yet.
"""
function add_point(o::OnlineLowess, x::Float64, y::Float64)
	result = ccall(
		(:jl_online_lowess_add_point, current_library()),
		Cdouble,
		(Ptr{Cvoid}, Cdouble, Cdouble),
		o.handle,
		x,
		y,
	)
	isnan(result) ? nothing : result
end

end # module FastLOWESS
