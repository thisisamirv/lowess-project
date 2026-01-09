"""
    fastLowess

High-performance LOWESS (Locally Weighted Scatterplot Smoothing) for Julia.

Provides bindings to the fastLowess Rust library for fast, robust LOWESS smoothing.

# Main Functions
- `smooth(x, y; kwargs...)`: Batch LOWESS smoothing
- `smooth_streaming(x, y; kwargs...)`: Streaming mode for large datasets
- `smooth_online(x, y; kwargs...)`: Online mode with sliding window

# Example
```julia
using fastLowess

x = collect(1.0:0.1:10.0)
y = sin.(x) .+ 0.1 .* randn(length(x))

result = smooth(x, y, fraction=0.3)
println("Smoothed values: ", result.y)
```
"""
module fastLowess

export smooth, smooth_streaming, smooth_online
export LowessResult, Diagnostics

const LIBNAME = Sys.iswindows() ? "fastlowess_jl.dll" : 
                Sys.isapple() ? "libfastlowess_jl.dylib" : "libfastlowess_jl.so"

# Try to find the library in common locations
function find_library()
    # Check environment variable first
    if haskey(ENV, "FASTLOWESS_LIB")
        return ENV["FASTLOWESS_LIB"]
    end
    
    # Check relative to this module (workspace structure)
    # Path: julia/src/fastLowess.jl -> julia/ -> bindings/julia/ -> bindings/ -> lowess-project/
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
    
    for path in candidates
        if isfile(path)
            return path
        end
    end
    
    # Fall back to system path
    return LIBNAME
end

const libfastlowess = find_library()

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
    return unsafe_wrap(Array, ptr, n, own=false) |> copy
end

function convert_result(c_result::CJlLowessResult)
    # Check for error
    if c_result.error != C_NULL
        error_msg = unsafe_string(c_result.error)
        # Free the result before throwing
        @ccall libfastlowess.jl_lowess_free_result(Ref(c_result)::Ptr{CJlLowessResult})::Cvoid
        error("fastLowess error: $error_msg")
    end
    
    n = Int(c_result.n)
    
    # Extract arrays
    x = ptr_to_vector(c_result.x, n)
    y = ptr_to_vector(c_result.y, n)
    
    if x === nothing || y === nothing
        @ccall libfastlowess.jl_lowess_free_result(Ref(c_result)::Ptr{CJlLowessResult})::Cvoid
        error("fastLowess error: result arrays are null")
    end
    
    standard_errors = ptr_to_vector(c_result.standard_errors, n)
    confidence_lower = ptr_to_vector(c_result.confidence_lower, n)
    confidence_upper = ptr_to_vector(c_result.confidence_upper, n)
    prediction_lower = ptr_to_vector(c_result.prediction_lower, n)
    prediction_upper = ptr_to_vector(c_result.prediction_upper, n)
    residuals = ptr_to_vector(c_result.residuals, n)
    robustness_weights = ptr_to_vector(c_result.robustness_weights, n)
    
    # Extract diagnostics
    diagnostics = if !isnan(c_result.rmse)
        Diagnostics(
            c_result.rmse,
            c_result.mae,
            c_result.r_squared,
            c_result.aic,
            c_result.aicc,
            c_result.effective_df,
            c_result.residual_sd
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
        c_result.fraction_used,
        Int(c_result.iterations_used),
        diagnostics
    )
    
    # Free the C result
    @ccall libfastlowess.jl_lowess_free_result(Ref(c_result)::Ptr{CJlLowessResult})::Cvoid
    
    return result
end

"""
    smooth(x, y; kwargs...) -> LowessResult

Perform LOWESS smoothing on the data using batch processing.

# Arguments
- `x::Vector{Float64}`: Independent variable values
- `y::Vector{Float64}`: Dependent variable values

# Keyword Arguments
- `fraction::Float64 = 0.67`: Smoothing fraction (proportion of data used for each fit)
- `iterations::Int = 3`: Number of robustness iterations
- `delta::Float64 = NaN`: Interpolation optimization threshold (NaN for auto)
- `weight_function::String = "tricube"`: Kernel function
  - Options: "tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle", "cosine"
- `robustness_method::String = "bisquare"`: Robustness method
  - Options: "bisquare", "huber", "talwar"
- `scaling_method::String = "mad"`: Scaling method for robustness
  - Options: "mad", "mar"
- `boundary_policy::String = "extend"`: Handling of edge effects
  - Options: "extend", "reflect", "zero", "noboundary"
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

# Returns
- `LowessResult`: Smoothed values and optional diagnostics

# Example
```julia
x = collect(1.0:0.1:10.0)
y = sin.(x) .+ 0.1 .* randn(length(x))

# Basic smoothing
result = smooth(x, y)

# With diagnostics
result = smooth(x, y, fraction=0.3, return_diagnostics=true)
println("R² = ", result.diagnostics.r_squared)
```
"""
function smooth(x::Vector{Float64}, y::Vector{Float64};
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
    parallel::Bool = true
)
    n = length(x)
    if n != length(y)
        throw(ArgumentError("x and y must have the same length"))
    end
    
    cv_ptr = isempty(cv_fractions) ? C_NULL : pointer(cv_fractions)
    cv_len = length(cv_fractions)
    
    c_result = @ccall libfastlowess.jl_lowess_smooth(
        x::Ptr{Cdouble},
        y::Ptr{Cdouble},
        Culong(n)::Culong,
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
        Cint(parallel)::Cint
    )::CJlLowessResult
    
    return convert_result(c_result)
end

"""
    smooth_streaming(x, y; kwargs...) -> LowessResult

Perform streaming LOWESS smoothing for large datasets.

Processes data in chunks to maintain constant memory usage.

# Arguments
- `x::Vector{Float64}`: Independent variable values
- `y::Vector{Float64}`: Dependent variable values

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

# Returns
- `LowessResult`: Smoothed values and optional diagnostics
"""
function smooth_streaming(x::Vector{Float64}, y::Vector{Float64};
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
    parallel::Bool = true
)
    n = length(x)
    if n != length(y)
        throw(ArgumentError("x and y must have the same length"))
    end
    
    c_result = @ccall libfastlowess.jl_lowess_streaming(
        x::Ptr{Cdouble},
        y::Ptr{Cdouble},
        Culong(n)::Culong,
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
        Cint(parallel)::Cint
    )::CJlLowessResult
    
    return convert_result(c_result)
end

"""
    smooth_online(x, y; kwargs...) -> LowessResult

Perform online LOWESS smoothing with a sliding window.

Maintains a sliding window for incremental updates.

# Arguments
- `x::Vector{Float64}`: Independent variable values
- `y::Vector{Float64}`: Dependent variable values

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

# Returns
- `LowessResult`: Smoothed values
"""
function smooth_online(x::Vector{Float64}, y::Vector{Float64};
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
    parallel::Bool = false
)
    n = length(x)
    if n != length(y)
        throw(ArgumentError("x and y must have the same length"))
    end
    
    c_result = @ccall libfastlowess.jl_lowess_online(
        x::Ptr{Cdouble},
        y::Ptr{Cdouble},
        Culong(n)::Culong,
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
        zero_weight_fallback::Cstring,
        Cint(parallel)::Cint
    )::CJlLowessResult
    
    return convert_result(c_result)
end

end # module
