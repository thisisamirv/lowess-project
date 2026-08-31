//! Java bindings for fastLowess (via JNI).
//!
//! Provides JNI-exported native methods, consumed by the
//! `com.thisisamirv.fastlowess.NativeBridge` Java class. Errors (both
//! ordinary failures and Rust panics) are surfaced as thrown
//! `java.lang.RuntimeException`s via [`jni::errors::ThrowRuntimeExAndDefault`],
//! since the JVM has native exception support (unlike C/Go).
#![allow(non_snake_case)]
// jni 0.22 deprecated several `Env` methods (e.g. `get_string`,
// `get_double_array_region`) in favor of newer per-type APIs
// (`JString::to_string`, `JDoubleArray::get_region`, etc). The deprecated
// methods still work identically; this binding uses them for brevity.
#![allow(deprecated)]

use fastLowess::internals::adapters::online::ParallelOnlineLowess;
use fastLowess::internals::adapters::streaming::ParallelStreamingLowess;
use fastLowess::internals::api::LowessBuilder;
use fastLowess::internals::binding_support as shared_parse;
use fastLowess::prelude::LowessResult;

use jni::EnvUnowned;
use jni::errors::ThrowRuntimeExAndDefault;
use jni::objects::{JClass, JDoubleArray, JObject, JString, JValue};
use jni::signature::MethodSignature;
use jni::strings::JNIStr;
use jni::sys::{jboolean, jdouble, jint, jlong};
use jni::{Env, errors::Error as JniError, jni_sig, jni_str};

const RESULT_CLASS: &JNIStr = jni_str!("com/thisisamirv/fastlowess/NativeResult");
const ONLINE_OUTPUT_CLASS: &JNIStr = jni_str!("com/thisisamirv/fastlowess/NativeOnlineOutput");
// Keep in sync with NativeResult's constructor parameter list.
const RESULT_CTOR_SIG: MethodSignature<'static, 'static> =
    jni_sig!("([D[D[D[D[D[D[D[D[D[DDIDDDDDDDZ)V");
// Keep in sync with NativeOnlineOutput's constructor parameter list.
const ONLINE_OUTPUT_CTOR_SIG: MethodSignature<'static, 'static> = jni_sig!("(ZDDDDI)V");

/// Error type used by every native method's `with_env` closure. Any
/// application-level failure (invalid arguments, runtime errors from the
/// fastLowess core) as well as any JNI-level failure are surfaced uniformly,
/// then thrown as a `java.lang.RuntimeException` by `ThrowRuntimeExAndDefault`.
#[derive(Debug)]
struct AppError(String);

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for AppError {}

impl From<String> for AppError {
    fn from(s: String) -> Self {
        AppError(s)
    }
}

impl From<&str> for AppError {
    fn from(s: &str) -> Self {
        AppError(s.to_string())
    }
}

impl From<JniError> for AppError {
    fn from(e: JniError) -> Self {
        AppError(e.to_string())
    }
}

type AppResult<T> = Result<T, AppError>;

fn jstring_to_string(env: &mut Env, s: &JString) -> Option<String> {
    if s.is_null() {
        return None;
    }
    env.get_string(s).ok().map(|s| s.into())
}

fn jstring_or_default(env: &mut Env, s: &JString, default: &str) -> String {
    jstring_to_string(env, s).unwrap_or_else(|| default.to_string())
}

fn jarray_len(env: &mut Env, arr: &JDoubleArray) -> usize {
    if arr.is_null() {
        0
    } else {
        env.get_array_length(arr).unwrap_or(0) as usize
    }
}

fn jarray_to_vec(env: &mut Env, arr: &JDoubleArray) -> Vec<f64> {
    let len = jarray_len(env, arr);
    if len == 0 {
        return Vec::new();
    }
    let mut buf = vec![0f64; len];
    if env.get_double_array_region(arr, 0, &mut buf).is_err() {
        return Vec::new();
    }
    buf
}

fn jarray_to_option_vec(env: &mut Env, arr: &JDoubleArray) -> Option<Vec<f64>> {
    let v = jarray_to_vec(env, arr);
    if v.is_empty() { None } else { Some(v) }
}

fn opt_f64(value: jdouble) -> Option<f64> {
    (!value.is_nan()).then_some(value)
}

// Converts a `Vec<f64>` (or `None`) into a Java `double[]` (or `null`).
fn vec_to_jdoublearray<'local>(
    env: &mut Env<'local>,
    data: &Option<Vec<f64>>,
) -> jni::errors::Result<JObject<'local>> {
    match data {
        None => Ok(JObject::null()),
        Some(v) => {
            let arr = env.new_double_array(v.len())?;
            env.set_double_array_region(&arr, 0, v)?;
            Ok(JObject::from(arr))
        }
    }
}

// Builds a `NativeResult` Java object from a `LowessResult<f64>`.
fn result_to_jobject<'local>(
    env: &mut Env<'local>,
    result: LowessResult<f64>,
) -> AppResult<JObject<'local>> {
    let (rmse, mae, r_squared, aic, aicc, effective_df, residual_sd) =
        shared_parse::extract_diagnostics(&result);
    let has_diagnostics = result.diagnostics.is_some();

    let x = vec_to_jdoublearray(env, &Some(result.x))?;
    let y = vec_to_jdoublearray(env, &Some(result.y))?;
    let standard_errors = vec_to_jdoublearray(env, &result.standard_errors)?;
    let confidence_lower = vec_to_jdoublearray(env, &result.confidence_lower)?;
    let confidence_upper = vec_to_jdoublearray(env, &result.confidence_upper)?;
    let prediction_lower = vec_to_jdoublearray(env, &result.prediction_lower)?;
    let prediction_upper = vec_to_jdoublearray(env, &result.prediction_upper)?;
    let residuals = vec_to_jdoublearray(env, &result.residuals)?;
    let robustness_weights = vec_to_jdoublearray(env, &result.robustness_weights)?;
    let cv_scores = vec_to_jdoublearray(env, &result.cv_scores)?;
    let fraction_used = result.fraction_used;
    let iterations_used = result.iterations_used.map(|i| i as jint).unwrap_or(-1);

    let class = env.find_class(RESULT_CLASS)?;
    let obj = env.new_object(
        class,
        RESULT_CTOR_SIG,
        &[
            JValue::Object(&x),
            JValue::Object(&y),
            JValue::Object(&standard_errors),
            JValue::Object(&confidence_lower),
            JValue::Object(&confidence_upper),
            JValue::Object(&prediction_lower),
            JValue::Object(&prediction_upper),
            JValue::Object(&residuals),
            JValue::Object(&robustness_weights),
            JValue::Object(&cv_scores),
            JValue::Double(fraction_used),
            JValue::Int(iterations_used),
            JValue::Double(rmse),
            JValue::Double(mae),
            JValue::Double(r_squared),
            JValue::Double(aic),
            JValue::Double(aicc),
            JValue::Double(effective_df),
            JValue::Double(residual_sd),
            JValue::Bool(has_diagnostics as jboolean),
        ],
    )?;
    Ok(obj)
}

/// Returns the crate version.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_version<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
) -> JString<'local> {
    env.with_env(|env| -> jni::errors::Result<JString<'local>> {
        env.new_string(env!("CARGO_PKG_VERSION"))
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// Returns true if this library was built with the `gpu` Cargo feature enabled.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_gpuEnabled(
    _env: EnvUnowned,
    _class: JClass,
) -> jboolean {
    cfg!(feature = "gpu") as jboolean
}

// ----------------------------------------------------------------------------
// Batch (Lowess)
// ----------------------------------------------------------------------------

// Opaque handle to a batch model.
struct JavaLowess {
    builder: Option<LowessBuilder<f64>>,
    cv_fractions: Option<Vec<f64>>,
    cv_method: Option<String>,
    cv_k: usize,
    cv_seed: Option<u64>,
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_lowessNew<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    fraction: jdouble,
    iterations: jint,
    delta: jdouble,
    weight_function: JString<'local>,
    robustness_method: JString<'local>,
    scaling_method: JString<'local>,
    boundary_policy: JString<'local>,
    confidence_intervals: jdouble,
    prediction_intervals: jdouble,
    return_diagnostics: jboolean,
    return_residuals: jboolean,
    return_robustness_weights: jboolean,
    zero_weight_fallback: JString<'local>,
    auto_converge: jdouble,
    cv_fractions: JDoubleArray<'local>,
    cv_method: JString<'local>,
    cv_k: jint,
    parallel: jboolean,
    return_se: jboolean,
    backend: JString<'local>,
) -> jlong {
    env.with_env(|env| -> AppResult<jlong> {
        let wf = jstring_or_default(env, &weight_function, shared_parse::DEFAULT_WEIGHT_FUNCTION);
        let rm = jstring_or_default(
            env,
            &robustness_method,
            shared_parse::DEFAULT_ROBUSTNESS_METHOD,
        );
        let sm = jstring_or_default(env, &scaling_method, shared_parse::DEFAULT_SCALING_METHOD);
        let bp = jstring_or_default(env, &boundary_policy, shared_parse::DEFAULT_BOUNDARY_POLICY);
        let zwf = jstring_or_default(
            env,
            &zero_weight_fallback,
            shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
        );
        let cv_method_str = jstring_or_default(env, &cv_method, "kfold");
        let backend_str = jstring_or_default(env, &backend, shared_parse::DEFAULT_BACKEND);
        let cv_fractions_vec = jarray_to_option_vec(env, &cv_fractions);

        let iterations = shared_parse::require_non_negative_usize("iterations", iterations)?;

        let builder = shared_parse::apply_builder_options(
            LowessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
                delta: opt_f64(delta),
                weight_function: Some(&wf),
                robustness_method: Some(&rm),
                zero_weight_fallback: Some(&zwf),
                boundary_policy: Some(&bp),
                scaling_method: Some(&sm),
                auto_converge: opt_f64(auto_converge),
                return_residuals,
                return_robustness_weights,
                return_diagnostics,
                return_se,
                confidence_intervals: opt_f64(confidence_intervals),
                prediction_intervals: opt_f64(prediction_intervals),
                parallel: Some(parallel),
                backend: Some(&backend_str),
                ..Default::default()
            },
        )?;

        Ok(Box::into_raw(Box::new(JavaLowess {
            builder: Some(builder),
            cv_fractions: cv_fractions_vec,
            cv_method: Some(cv_method_str),
            cv_k: cv_k.max(2) as usize,
            cv_seed: None,
        })) as jlong)
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_lowessSetCvSeed(
    _env: EnvUnowned,
    _class: JClass,
    handle: jlong,
    seed: jlong,
) {
    if handle != 0 {
        let lowess = unsafe { &mut *(handle as *mut JavaLowess) };
        lowess.cv_seed = Some(seed as u64);
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_lowessFit<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    x: JDoubleArray<'local>,
    y: JDoubleArray<'local>,
    custom_weights: JDoubleArray<'local>,
) -> JObject<'local> {
    env.with_env(|env| -> AppResult<JObject<'local>> {
        if handle == 0 {
            return Err(shared_parse::MODEL_POINTER_IS_NULL.into());
        }
        let lowess = unsafe { &mut *(handle as *mut JavaLowess) };
        let x_vec = jarray_to_vec(env, &x);
        let y_vec = jarray_to_vec(env, &y);
        if x_vec.is_empty() || y_vec.is_empty() {
            return Err(shared_parse::INVALID_DATA_INPUTS.into());
        }
        let cw = jarray_to_option_vec(env, &custom_weights);

        let Some(mut builder) = lowess.builder.clone() else {
            return Err(shared_parse::MODEL_NOT_INITIALIZED.into());
        };
        builder = shared_parse::apply_cross_validation(
            builder,
            lowess.cv_fractions.as_deref(),
            lowess.cv_method.as_deref(),
            Some(lowess.cv_k),
            lowess.cv_seed,
        )?;
        let model = shared_parse::build_batch(builder, cw).map_err(|e| e.message)?;
        let result = model.fit(&x_vec, &y_vec).map_err(|e| e.to_string())?;
        result_to_jobject(env, result)
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_lowessFree(
    _env: EnvUnowned,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut JavaLowess)) };
    }
}

// ----------------------------------------------------------------------------
// Streaming (StreamingLowess)
// ----------------------------------------------------------------------------

struct JavaStreamingLowess {
    model: ParallelStreamingLowess<f64>,
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_streamingNew<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    fraction: jdouble,
    iterations: jint,
    delta: jdouble,
    weight_function: JString<'local>,
    robustness_method: JString<'local>,
    scaling_method: JString<'local>,
    boundary_policy: JString<'local>,
    return_diagnostics: jboolean,
    return_residuals: jboolean,
    return_robustness_weights: jboolean,
    zero_weight_fallback: JString<'local>,
    auto_converge: jdouble,
    parallel: jboolean,
    chunk_size: jint,
    overlap: jint,
    merge_strategy: JString<'local>,
    confidence_intervals: jdouble,
    prediction_intervals: jdouble,
) -> jlong {
    env.with_env(|env| -> AppResult<jlong> {
        let wf = jstring_or_default(env, &weight_function, shared_parse::DEFAULT_WEIGHT_FUNCTION);
        let rm = jstring_or_default(
            env,
            &robustness_method,
            shared_parse::DEFAULT_ROBUSTNESS_METHOD,
        );
        let sm = jstring_or_default(env, &scaling_method, shared_parse::DEFAULT_SCALING_METHOD);
        let bp = jstring_or_default(env, &boundary_policy, shared_parse::DEFAULT_BOUNDARY_POLICY);
        let zwf = jstring_or_default(
            env,
            &zero_weight_fallback,
            shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
        );
        let ms = jstring_or_default(
            env,
            &merge_strategy,
            shared_parse::DEFAULT_STREAMING_MERGE_STRATEGY,
        );

        let chunk_size = shared_parse::require_positive_usize("chunkSize", chunk_size)?;

        let builder = shared_parse::apply_builder_options(
            LowessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations as usize),
                delta: opt_f64(delta),
                weight_function: Some(&wf),
                robustness_method: Some(&rm),
                zero_weight_fallback: Some(&zwf),
                boundary_policy: Some(&bp),
                scaling_method: Some(&sm),
                auto_converge: opt_f64(auto_converge),
                return_residuals,
                return_robustness_weights,
                return_diagnostics,
                confidence_intervals: opt_f64(confidence_intervals),
                prediction_intervals: opt_f64(prediction_intervals),
                parallel: Some(parallel),
                ..Default::default()
            },
        )?;

        let model = shared_parse::build_streaming(
            builder,
            Some(chunk_size),
            (overlap >= 0).then_some(overlap as usize),
            Some(&ms),
        )
        .map_err(|e| e.message)?;

        Ok(Box::into_raw(Box::new(JavaStreamingLowess { model })) as jlong)
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_streamingProcess<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    x: JDoubleArray<'local>,
    y: JDoubleArray<'local>,
) -> JObject<'local> {
    env.with_env(|env| -> AppResult<JObject<'local>> {
        if handle == 0 {
            return Err(shared_parse::MODEL_POINTER_IS_NULL.into());
        }
        let streaming = unsafe { &mut *(handle as *mut JavaStreamingLowess) };
        let x_vec = jarray_to_vec(env, &x);
        let y_vec = jarray_to_vec(env, &y);
        if x_vec.is_empty() || y_vec.is_empty() {
            return Err(shared_parse::INVALID_DATA_INPUTS.into());
        }
        let result = streaming
            .model
            .process_chunk(&x_vec, &y_vec)
            .map_err(|e| e.to_string())?;
        result_to_jobject(env, result)
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_streamingFinalize<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> JObject<'local> {
    env.with_env(|env| -> AppResult<JObject<'local>> {
        if handle == 0 {
            return Err(shared_parse::MODEL_POINTER_IS_NULL.into());
        }
        let streaming = unsafe { &mut *(handle as *mut JavaStreamingLowess) };
        let result = streaming.model.finalize().map_err(|e| e.to_string())?;
        result_to_jobject(env, result)
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_streamingFree(
    _env: EnvUnowned,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut JavaStreamingLowess)) };
    }
}

// ----------------------------------------------------------------------------
// Online (OnlineLowess)
// ----------------------------------------------------------------------------

struct JavaOnlineLowess {
    model: ParallelOnlineLowess<f64>,
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_onlineNew<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    fraction: jdouble,
    iterations: jint,
    delta: jdouble,
    weight_function: JString<'local>,
    robustness_method: JString<'local>,
    scaling_method: JString<'local>,
    boundary_policy: JString<'local>,
    return_robustness_weights: jboolean,
    return_diagnostics: jboolean,
    return_residuals: jboolean,
    zero_weight_fallback: JString<'local>,
    auto_converge: jdouble,
    parallel: jboolean,
    window_capacity: jint,
    min_points: jint,
    update_mode: JString<'local>,
    confidence_intervals: jdouble,
    prediction_intervals: jdouble,
) -> jlong {
    env.with_env(|env| -> AppResult<jlong> {
        let wf = jstring_or_default(env, &weight_function, shared_parse::DEFAULT_WEIGHT_FUNCTION);
        let rm = jstring_or_default(
            env,
            &robustness_method,
            shared_parse::DEFAULT_ROBUSTNESS_METHOD,
        );
        let sm = jstring_or_default(env, &scaling_method, shared_parse::DEFAULT_SCALING_METHOD);
        let bp = jstring_or_default(env, &boundary_policy, shared_parse::DEFAULT_BOUNDARY_POLICY);
        let zwf = jstring_or_default(
            env,
            &zero_weight_fallback,
            shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
        );
        let um = jstring_or_default(env, &update_mode, shared_parse::DEFAULT_ONLINE_UPDATE_MODE);

        let window_capacity =
            shared_parse::require_positive_usize("windowCapacity", window_capacity)?;
        let min_points = shared_parse::require_positive_usize("minPoints", min_points)?;

        let builder = shared_parse::apply_builder_options(
            LowessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations as usize),
                delta: opt_f64(delta),
                weight_function: Some(&wf),
                robustness_method: Some(&rm),
                zero_weight_fallback: Some(&zwf),
                boundary_policy: Some(&bp),
                scaling_method: Some(&sm),
                auto_converge: opt_f64(auto_converge),
                return_residuals,
                return_robustness_weights,
                return_diagnostics,
                confidence_intervals: opt_f64(confidence_intervals),
                prediction_intervals: opt_f64(prediction_intervals),
                parallel: Some(parallel),
                ..Default::default()
            },
        )?;

        let model =
            shared_parse::build_online(builder, Some(window_capacity), Some(min_points), Some(&um))
                .map_err(|e| e.message)?;

        Ok(Box::into_raw(Box::new(JavaOnlineLowess { model })) as jlong)
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_onlineAddPoint<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    x: jdouble,
    y: jdouble,
) -> JObject<'local> {
    env.with_env(|env| -> AppResult<JObject<'local>> {
        if handle == 0 {
            return Err(shared_parse::MODEL_POINTER_IS_NULL.into());
        }
        let online = unsafe { &mut *(handle as *mut JavaOnlineLowess) };
        let point = online.model.add_point(x, y).map_err(|e| e.to_string())?;

        let (has_value, y_val, standard_error, residual, robustness_weight, iterations_used) =
            match point {
                None => (false, f64::NAN, f64::NAN, f64::NAN, f64::NAN, -1),
                Some(o) => {
                    let (se, res, rw, iters) = shared_parse::extract_online_output(&o);
                    (true, o.y, se, res, rw, iters)
                }
            };

        let class = env.find_class(ONLINE_OUTPUT_CLASS)?;
        let obj = env.new_object(
            class,
            ONLINE_OUTPUT_CTOR_SIG,
            &[
                JValue::Bool(has_value as jboolean),
                JValue::Double(y_val),
                JValue::Double(standard_error),
                JValue::Double(residual),
                JValue::Double(robustness_weight),
                JValue::Int(iterations_used),
            ],
        )?;
        Ok(obj)
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_thisisamirv_fastlowess_NativeBridge_onlineFree(
    _env: EnvUnowned,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut JavaOnlineLowess)) };
    }
}
