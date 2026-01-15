/**
 * @file batch_smoothing.cpp
 * @brief fastlowess Batch Smoothing Example
 *
 * This example demonstrates batch LOWESS smoothing features:
 * - Basic smoothing with different parameters
 * - Robustness iterations for outlier handling
 * - Confidence and prediction intervals
 * - Diagnostics and cross-validation
 *
 * The Lowess class is the primary interface for
 * processing complete datasets that fit in memory.
 */

#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

#include "fastlowess.hpp"


// Synthetic data generation
struct Data {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> y_true;
};

Data generate_sample_data(size_t n_points = 100) {
  Data data;
  data.x.resize(n_points);
  data.y.resize(n_points);
  data.y_true.resize(n_points);

  std::mt19937 gen(42);
  std::normal_distribution<> noise(0, 1.5);
  // Random number generator for outliers
  std::uniform_real_distribution<> outlier_mag(10.0, 20.0);
  std::uniform_int_distribution<> outlier_sign(0, 1);

  for (size_t i = 0; i < n_points; ++i) {
    data.x[i] = static_cast<double>(i) * 50.0 / static_cast<double>(n_points - 1);
    
    // Trend + Seasonality
    data.y_true[i] = 0.5 * data.x[i] + 5.0 * std::sin(data.x[i] * 0.5);
    
    // Add noise
    data.y[i] = data.y_true[i] + noise(gen);
  }

  // Add significant outliers (10%)
  size_t n_outliers = n_points / 10;
  std::uniform_int_distribution<> idx_dist(0, n_points - 1);
  
  for(size_t k=0; k<n_outliers; ++k) {
      size_t idx = idx_dist(gen);
      double val = outlier_mag(gen);
      if(outlier_sign(gen) == 0) val = -val;
      data.y[idx] += val;
  }

  return data;
}

int main() {
  std::cout << "=== fastlowess Batch Smoothing Example ===" << std::endl;

  // 1. Generate Data
  auto data = generate_sample_data(100);
  std::cout << "Generated " << data.x.size() << " data points with outliers." << std::endl;

  try {
    // 2. Basic Smoothing (Default parameters)
    std::cout << "Running basic smoothing..." << std::endl;
    fastlowess::LowessOptions basic_opts;
    basic_opts.fraction = 0.05;
    basic_opts.iterations = 0;
    fastlowess::Lowess model_basic(basic_opts);
    auto res_basic = model_basic.fit(data.x, data.y);

    // 3. Robust Smoothing (IRLS)
    std::cout << "Running robust smoothing (3 iterations)..." << std::endl;
    fastlowess::LowessOptions robust_opts;
    robust_opts.fraction = 0.05;
    robust_opts.iterations = 3;
    robust_opts.robustness_method = "bisquare";
    robust_opts.return_robustness_weights = true;
    
    fastlowess::Lowess model_robust(robust_opts);
    auto res_robust = model_robust.fit(data.x, data.y);
    
    // 4. Uncertainty Quantification
    std::cout << "Computing confidence and prediction intervals..." << std::endl;
    fastlowess::LowessOptions interval_opts;
    interval_opts.fraction = 0.05;
    interval_opts.confidence_intervals = 0.95;
    interval_opts.prediction_intervals = 0.95;
    interval_opts.return_diagnostics = true;
    
    fastlowess::Lowess model_intervals(interval_opts);
    auto res_intervals = model_intervals.fit(data.x, data.y);

    // 5. Cross-Validation for optimal fraction
    std::cout << "Running cross-validation to find optimal fraction..." << std::endl;
    
    // Manual CV search
    std::vector<double> fractions = {0.05, 0.1, 0.2, 0.4};
    double best_fraction = 0.0;
    double min_rmse = 1e9;
    
    for(double f : fractions) {
        fastlowess::LowessOptions cv_opts;
        cv_opts.fraction = f;
        cv_opts.return_diagnostics = true;
        fastlowess::Lowess model(cv_opts);
        auto res = model.fit(data.x, data.y);
        if(res.diagnostics().has_value()) {
             double rmse = res.diagnostics().rmse;
             if(rmse < min_rmse) {
                 min_rmse = rmse;
                 best_fraction = f;
             }
        }
    }
    std::cout << "Optimal fraction found (manual CV): " << best_fraction << std::endl;

    // Diagnostics Printout
    if (res_intervals.diagnostics().has_value()) {
      auto diag = res_intervals.diagnostics();
      std::cout << "\nFit Statistics (Intervals Model):" << std::endl;
      std::cout << " - R^2:   " << diag.r_squared << std::endl;
      std::cout << " - RMSE: " << diag.rmse << std::endl;
      std::cout << " - MAE:  " << diag.mae << std::endl;
    }

    // 6. Boundary Policy Comparison
    std::cout << "\nDemonstrating boundary policy effects on linear data..." << std::endl;
    std::vector<double> xl(50), yl(50);
    for(size_t i=0; i<50; ++i) {
        xl[i] = static_cast<double>(i) * 10.0 / 49.0;
        yl[i] = 2.0 * xl[i] + 1.0;
    }

    fastlowess::LowessOptions opt_ext;
    opt_ext.fraction = 0.6;
    opt_ext.boundary_policy = "extend";
    auto r_ext = fastlowess::Lowess(opt_ext).fit(xl, yl);
    
    fastlowess::LowessOptions opt_ref;
    opt_ref.fraction = 0.6;
    opt_ref.boundary_policy = "reflect";
    auto r_ref = fastlowess::Lowess(opt_ref).fit(xl, yl);
    
    fastlowess::LowessOptions opt_zero;
    opt_zero.fraction = 0.6;
    opt_zero.boundary_policy = "zero";
    auto r_zr = fastlowess::Lowess(opt_zero).fit(xl, yl);

    std::cout << "Boundary policy comparison:" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << " - Extend (Default): first=" << r_ext.y(0) << ", last=" << r_ext.y(49) << std::endl;
    std::cout << " - Reflect:          first=" << r_ref.y(0) << ", last=" << r_ref.y(49) << std::endl;
    std::cout << " - Zero:             first=" << r_zr.y(0)  << ", last=" << r_zr.y(49)  << std::endl;

  } catch (const fastlowess::LowessError &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n=== Batch Smoothing Example Complete ===" << std::endl;
  return 0;
}
