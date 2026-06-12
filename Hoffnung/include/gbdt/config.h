#pragma once
#include <string>
#include <vector>

namespace gbdt {

struct GBDTConfig {
    // Tree building
    int max_depth = 6;
    int min_samples_leaf = 10;
    float min_gain_to_split = 0.0f;
    int max_bins = 256;

    // Regularization
    float lambda_l2 = 1.0f;    // L2 on leaf values
    float lambda_l1 = 0.0f;    // L1 on leaf values
    float gamma = 0.0f;        // Split complexity penalty

    // Boosting
    int num_trees = 100;
    float learning_rate = 0.1f;
    float subsample_row = 1.0f;  // row subsample ratio
    float subsample_col = 1.0f;  // column subsample ratio

    // Loss
    std::string loss_type = "mse";  // "mse", "huber", "quantile", "rankic", "custom"
    float huber_delta = 1.0f;       // Huber loss transition point
    float mae_eps = 1e-8f;          // MAE Hessian stabilizer (non-zero for GBDT)

    // Early stopping
    int early_stopping_rounds = 10;
    float early_stopping_tol = 1e-4f;

    // Line search
    bool use_line_search = false;

    // Missing values
    bool enable_missing_values = false;

    // Threading
    int num_threads = 1;  // OpenMP threads

    // Seed
    int random_seed = 42;
};

struct MonotoneConstraint {
    int feature_idx;
    enum Direction { INCREASING, DECREASING, NONE } dir = NONE;
};

}  // namespace gbdt
