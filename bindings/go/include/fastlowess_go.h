typedef struct fastlowess_GoLowess fastlowess_GoLowess;

typedef struct fastlowess_GoOnlineLowess fastlowess_GoOnlineLowess;

typedef struct fastlowess_GoStreamingLowess fastlowess_GoStreamingLowess;

/**
 * Result struct that can be passed across FFI boundary.
 * All arrays are allocated by Rust and must be freed by Rust.
 */
typedef struct fastlowess_GoLowessResult {
  /**
   * Sorted x values (length = n)
   */
  double *x;
  /**
   * Smoothed y values (length = n)
   */
  double *y;
  /**
   * Number of data points
   */
  unsigned long n;
  /**
   * Standard errors (NULL if not computed)
   */
  double *standard_errors;
  /**
   * Lower confidence bounds (NULL if not computed)
   */
  double *confidence_lower;
  /**
   * Upper confidence bounds (NULL if not computed)
   */
  double *confidence_upper;
  /**
   * Lower prediction bounds (NULL if not computed)
   */
  double *prediction_lower;
  /**
   * Upper prediction bounds (NULL if not computed)
   */
  double *prediction_upper;
  /**
   * Residuals (NULL if not computed)
   */
  double *residuals;
  /**
   * Robustness weights (NULL if not computed)
   */
  double *robustness_weights;
  /**
   * Cross-validation scores (NULL if not computed, length = cv_scores_len)
   */
  double *cv_scores;
  /**
   * Number of cross-validation scores
   */
  unsigned long cv_scores_len;
  /**
   * Fraction used for smoothing
   */
  double fraction_used;
  /**
   * Number of iterations performed (-1 if not available)
   */
  int iterations_used;
  /**
   * Diagnostics (NaN if not computed)
   */
  double rmse;
  double mae;
  double r_squared;
  double aic;
  double aicc;
  double effective_df;
  double residual_sd;
  /**
   * Error message (NULL if no error)
   */
  char *error;
} fastlowess_GoLowessResult;

typedef struct fastlowess_GoOnlineOutput {
  int has_value;
  double y;
  double standard_error;
  double residual;
  double robustness_weight;
  int iterations_used;
  char *error;
} fastlowess_GoOnlineOutput;

const char *go_last_error_message(void);

/**
 * Returns 1 if this library was built with the `gpu` Cargo feature enabled, 0 otherwise.
 */
int go_gpu_enabled(void);

/**
 * Go wrapper constructor.
 *
 * # Safety
 * Pointers must be valid null-terminated strings or null. Arrays must be valid.
 */
struct fastlowess_GoLowess *go_lowess_new(double fraction,
                                          int iterations,
                                          double delta,
                                          const char *weight_function,
                                          const char *robustness_method,
                                          const char *scaling_method,
                                          const char *boundary_policy,
                                          double confidence_intervals,
                                          double prediction_intervals,
                                          int return_diagnostics,
                                          int return_residuals,
                                          int return_robustness_weights,
                                          const char *zero_weight_fallback,
                                          double auto_converge,
                                          const double *cv_fractions,
                                          unsigned long cv_fractions_len,
                                          const char *cv_method,
                                          int cv_k,
                                          int parallel,
                                          int return_se,
                                          const char *backend);

/**
 * Set CV seed for reproducible K-fold splits.
 *
 * # Safety
 * ptr must be valid.
 */
void go_lowess_set_cv_seed(struct fastlowess_GoLowess *ptr, unsigned long seed);

/**
 * Fit the batch model.
 *
 * # Safety
 * `ptr` must be a valid GoLowess pointer. `x_values` and `y_values` must be
 * valid arrays of length `n`. `custom_weights` is optional: pass null and 0 to omit.
 */
struct fastlowess_GoLowessResult go_lowess_fit(struct fastlowess_GoLowess *ptr,
                                               const double *x_values,
                                               const double *y_values,
                                               unsigned long n,
                                               const double *custom_weights,
                                               unsigned long custom_weights_len);

/**
 * Free batch model.
 *
 * # Safety
 * `ptr` must be a valid pointer returned by `go_lowess_new` or null.
 */
void go_lowess_free(struct fastlowess_GoLowess *ptr);

/**
 * Create a new Streaming Lowess model.
 *
 * # Safety
 * Pointers must be valid null-terminated strings or null.
 */
struct fastlowess_GoStreamingLowess *go_streaming_new(double fraction,
                                                      int iterations,
                                                      double delta,
                                                      const char *weight_function,
                                                      const char *robustness_method,
                                                      const char *scaling_method,
                                                      const char *boundary_policy,
                                                      int return_diagnostics,
                                                      int return_residuals,
                                                      int return_robustness_weights,
                                                      const char *zero_weight_fallback,
                                                      double auto_converge,
                                                      int parallel,
                                                      int chunk_size,
                                                      int overlap,
                                                      const char *merge_strategy,
                                                      double confidence_intervals,
                                                      double prediction_intervals);

/**
 * Process a chunk of data.
 *
 * # Safety
 * `ptr` must be valid. `x_values` and `y_values` must be valid arrays of
 * length `n`.
 */
struct fastlowess_GoLowessResult go_streaming_process(struct fastlowess_GoStreamingLowess *ptr,
                                                      const double *x_values,
                                                      const double *y_values,
                                                      unsigned long n);

/**
 * Finalize the streaming process.
 *
 * # Safety
 * `ptr` must be valid.
 */
struct fastlowess_GoLowessResult go_streaming_finalize(struct fastlowess_GoStreamingLowess *ptr);

/**
 * Free streaming model.
 *
 * # Safety
 * `ptr` must be valid or null.
 */
void go_streaming_free(struct fastlowess_GoStreamingLowess *ptr);

/**
 * Create a new Online Lowess model.
 *
 * # Safety
 * Pointers must be valid null-terminated strings or null.
 */
struct fastlowess_GoOnlineLowess *go_online_new(double fraction,
                                                int iterations,
                                                double delta,
                                                const char *weight_function,
                                                const char *robustness_method,
                                                const char *scaling_method,
                                                const char *boundary_policy,
                                                int return_robustness_weights,
                                                int return_diagnostics,
                                                int return_residuals,
                                                const char *zero_weight_fallback,
                                                double auto_converge,
                                                int parallel,
                                                int window_capacity,
                                                int min_points,
                                                const char *update_mode,
                                                double confidence_intervals,
                                                double prediction_intervals);

/**
 * Add a single point to the model and return its smoothed value.
 * `has_value = 0` in the result means the window is still filling.
 *
 * # Safety
 * `ptr` must be a valid `GoOnlineLowess` pointer.
 */
struct fastlowess_GoOnlineOutput go_online_add_point(struct fastlowess_GoOnlineLowess *ptr,
                                                     double x,
                                                     double y);

/**
 * Free the error string in a GoOnlineOutput (call only when error != NULL).
 *
 * # Safety
 * `output` must be a valid pointer and `output->error` must have been allocated by Rust.
 */
void go_online_free_output(struct fastlowess_GoOnlineOutput *output);

/**
 * Free online model.
 *
 * # Safety
 * `ptr` must be valid or null.
 */
void go_online_free(struct fastlowess_GoOnlineLowess *ptr);

/**
 * Free a GoLowessResult.
 *
 * # Safety
 * `result` must be a valid pointer to a GoLowessResult struct.
 */
void go_lowess_free_result(struct fastlowess_GoLowessResult *result);
