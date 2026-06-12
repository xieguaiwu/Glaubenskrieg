#include "gbdt/gbdt.h"
#include <gbdt/json_utils.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdexcept>

namespace gbdt {

float compute_scalar_loss(const torch::Tensor& y_pred, const torch::Tensor& y_true,
                          const std::string& loss_type,
                          float huber_delta) {
    auto diff = y_pred - y_true;
    if (loss_type == "mse") {
        return diff.pow(2).mean().item<float>();
    } else if (loss_type == "mae") {
        return torch::abs(diff).mean().item<float>();
    } else if (loss_type == "huber") {
        float delta = huber_delta;
        auto abs_diff = torch::abs(diff);
        auto small_mask = abs_diff <= delta;
        auto huber_loss = torch::where(
            small_mask,
            0.5f * diff.pow(2),
            delta * (abs_diff - 0.5f * delta));
        return huber_loss.mean().item<float>();
    } else {
        throw std::invalid_argument("Unsupported loss_type '" + loss_type + "'.");
    }
}

GradHessOutput GradientBridge::compute_gradients_hessians(
    const torch::Tensor& y_true,
    const torch::Tensor& y_pred,
    const std::string& loss_type,
    const torch::Tensor& loss_params)
{
    // Extract loss parameters from loss_params tensor: [0]=huber_delta, [1]=mae_eps
    float huber_delta = 1.0f;
    float mae_eps = 1e-8f;
    if (loss_params.defined() && loss_params.numel() >= 2) {
        auto lp = loss_params.accessor<float, 1>();
        huber_delta = lp[0];
        mae_eps = lp[1];
    }

    GradHessOutput out;
    auto diff = y_pred - y_true;

    if (loss_type == "mse") {
        out.gradients = 2.0f * diff;
        out.hessians = 2.0f * torch::ones_like(y_true);
    } else if (loss_type == "mae") {
        out.gradients = torch::sign(diff);
        out.hessians = torch::full_like(y_true, mae_eps);
    } else if (loss_type == "huber") {
        float delta = huber_delta;
        auto abs_diff = torch::abs(diff);
        auto small_mask = abs_diff <= delta;
        out.gradients = torch::where(
            small_mask, diff, delta * torch::sign(diff));
        out.hessians = torch::where(
            small_mask,
            torch::ones_like(y_true),
            torch::full_like(y_true, mae_eps));
    } else {
        throw std::invalid_argument(
            "Unsupported loss_type '" + loss_type + "'. "
            "Supported types: 'mse', 'mae', 'huber'. "
            "For custom losses, use fit_with_grad_hess() to pass "
            "pre-computed gradients and hessians.");
    }

    out.loss_value = compute_scalar_loss(y_pred, y_true, loss_type, huber_delta);
    return out;
}

GBDT::GBDT(const GBDTConfig& config)
    : config_(config)
    , init_pred_(0.0f)
    , rng_(config.random_seed)
{
    tree_builder_ = std::make_unique<TreeBuilder>(config);
    bridge_ = std::make_unique<GradientBridge>();
}

std::vector<int> GBDT::random_subset(int n, float ratio) {
    if (n <= 0) return {};

    if (ratio <= 0.0f || ratio >= 1.0f) {
        std::vector<int> result(n);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }

    int k = std::max(1, static_cast<int>(n * ratio));
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    if (k >= n / 2) {
        std::shuffle(indices.begin(), indices.end(), rng_);
    } else {
        for (int i = 0; i < k; ++i) {
            int j = i + std::uniform_int_distribution<int>(0, n - i - 1)(rng_);
            std::swap(indices[i], indices[j]);
        }
    }
    indices.resize(k);
    return indices;
}

std::vector<int> GBDT::random_col_subset(int n, float ratio) {
    if (n <= 0) return {};

    if (ratio <= 0.0f || ratio >= 1.0f) {
        std::vector<int> result(n);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }

    int k = std::max(1, static_cast<int>(n * ratio));
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    if (k >= n / 2) {
        std::shuffle(indices.begin(), indices.end(), rng_);
    } else {
        for (int i = 0; i < k; ++i) {
            int j = i + std::uniform_int_distribution<int>(0, n - i - 1)(rng_);
            std::swap(indices[i], indices[j]);
        }
    }
    indices.resize(k);
    std::sort(indices.begin(), indices.end());
    return indices;
}

std::vector<TreeNode> GBDT::fit_one_tree(
    const torch::Tensor& X,
    const torch::Tensor& gradients,
    const torch::Tensor& hessians,
    const std::vector<MonotoneConstraint>& constraints)
{
    int64_t N = X.size(0);
    int64_t D = X.size(1);

    torch::Tensor X_sub = X;
    torch::Tensor grad_sub = gradients;
    torch::Tensor hess_sub = hessians;

    if (config_.subsample_row < 1.0f) {
        auto row_idx = random_subset(static_cast<int>(N), config_.subsample_row);
        auto row_idx_t = torch::tensor(row_idx, torch::kInt64);
        X_sub = X.index_select(0, row_idx_t);
        grad_sub = gradients.index_select(0, row_idx_t);
        hess_sub = hessians.index_select(0, row_idx_t);
    }

    std::vector<int> col_idx;
    if (config_.subsample_col < 1.0f) {
        col_idx = random_col_subset(static_cast<int>(D), config_.subsample_col);
        auto col_idx_t = torch::tensor(col_idx, torch::kInt64);
        X_sub = X_sub.index_select(1, col_idx_t);
    }

    auto tree = tree_builder_->build_tree(X_sub, grad_sub, hess_sub, constraints);

    if (!col_idx.empty()) {
        for (auto& node : tree) {
            if (!node.is_leaf() &&
                node.feature_idx >= 0 &&
                node.feature_idx < static_cast<int>(col_idx.size())) {
                node.feature_idx = col_idx[node.feature_idx];
            }
        }
    }

    return tree;
}

float GBDT::line_search(
    const torch::Tensor& gradients,
    const torch::Tensor& hessians,
    const torch::Tensor& tree_pred) const
{
    float num = -torch::sum(gradients * tree_pred).item<float>();
    float den = torch::sum(hessians * tree_pred * tree_pred).item<float>();

    if (den < 1e-10f) return 1.0f;
    return num / den;
}

torch::Tensor GBDT::predict_tree(
    const std::vector<TreeNode>& tree,
    const torch::Tensor& X,
    const std::vector<int>& col_idx) const
{
    int64_t N = X.size(0);

    if (tree.empty()) {
        return torch::zeros({N}, torch::kFloat32);
    }

    auto result = torch::empty({N}, torch::kFloat32);
    auto result_acc = result.accessor<float, 1>();

    auto X_contig = X.contiguous();
    const float* x_data = X_contig.const_data_ptr<float>();
    int64_t num_cols = X_contig.size(1);

    bool use_col_map = !col_idx.empty();
    int num_col_map = static_cast<int>(col_idx.size());

    for (int64_t s = 0; s < N; ++s) {
        int node = 0;
        while (!tree[node].is_leaf()) {
            int feat = tree[node].feature_idx;
            if (feat < 0) break;

            int actual_feat = feat;
            if (use_col_map) {
                if (feat >= num_col_map) {
                    node = tree[node].default_left
                               ? tree[node].left_child
                               : tree[node].right_child;
                    continue;
                }
                actual_feat = col_idx[feat];
            }

            float val = x_data[s * num_cols + actual_feat];

            if (config_.enable_missing_values && std::isnan(val)) {
                node = tree[node].default_left
                           ? tree[node].left_child
                           : tree[node].right_child;
            } else if (val <= tree[node].split_value) {
                node = tree[node].left_child;
            } else {
                node = tree[node].right_child;
            }
        }

        result_acc[s] = (node >= 0 && node < static_cast<int>(tree.size()))
                            ? tree[node].leaf_value
                            : 0.0f;
    }

    return result;
}

torch::Tensor GBDT::predict_batch(const torch::Tensor& X) const {
    auto flat_trees = InferenceEngine::convert_forest(trees_);
    return InferenceEngine::batch_predict(
        X, flat_trees, init_pred_, tree_step_sizes_);
}

torch::Tensor GBDT::predict(const torch::Tensor& X) const {
    int64_t N = X.size(0);
    auto y_pred = torch::full({N}, init_pred_, torch::kFloat32);

    for (size_t i = 0; i < trees_.size(); ++i) {
        float step = (i < tree_step_sizes_.size())
                         ? tree_step_sizes_[i]
                         : config_.learning_rate;
        auto tree_pred = predict_tree(trees_[i], X);
        y_pred += step * tree_pred;
    }

    return y_pred;
}

std::string GBDT::to_json() const {
    std::ostringstream oss;
    oss << std::setprecision(9);
    JsonWriter w(oss);

    oss << "{";

    w.write_string("init_pred");
    oss << ":" << init_pred_ << ",";

    w.write_string("config");
    oss << ":{";
    w.write_string("max_depth"); oss << ":" << config_.max_depth << ",";
    w.write_string("min_samples_leaf"); oss << ":" << config_.min_samples_leaf << ",";
    w.write_string("min_gain_to_split"); oss << ":"; w.write_float(config_.min_gain_to_split); oss << ",";
    w.write_string("max_bins"); oss << ":" << config_.max_bins << ",";
    w.write_string("lambda_l2"); oss << ":"; w.write_float(config_.lambda_l2); oss << ",";
    w.write_string("lambda_l1"); oss << ":"; w.write_float(config_.lambda_l1); oss << ",";
    w.write_string("gamma"); oss << ":"; w.write_float(config_.gamma); oss << ",";
    w.write_string("num_trees"); oss << ":" << config_.num_trees << ",";
    w.write_string("learning_rate"); oss << ":"; w.write_float(config_.learning_rate); oss << ",";
    w.write_string("subsample_row"); oss << ":"; w.write_float(config_.subsample_row); oss << ",";
    w.write_string("subsample_col"); oss << ":"; w.write_float(config_.subsample_col); oss << ",";
    w.write_string("loss_type"); oss << ":"; w.write_string(config_.loss_type); oss << ",";
    w.write_string("early_stopping_rounds"); oss << ":" << config_.early_stopping_rounds << ",";
    w.write_string("early_stopping_tol"); oss << ":"; w.write_float(config_.early_stopping_tol); oss << ",";
    w.write_string("use_line_search"); oss << ":"; w.write_bool(config_.use_line_search); oss << ",";
    w.write_string("enable_missing_values"); oss << ":"; w.write_bool(config_.enable_missing_values); oss << ",";
    w.write_string("huber_delta"); oss << ":"; w.write_float(config_.huber_delta); oss << ",";
    w.write_string("mae_eps"); oss << ":"; w.write_float(config_.mae_eps);
    oss << "},";

    w.write_string("trees");
    oss << ":[";
    for (size_t i = 0; i < trees_.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "[";
        for (size_t j = 0; j < trees_[i].size(); ++j) {
            if (j > 0) oss << ",";
            const auto& n = trees_[i][j];
            oss << "{";
            w.write_string("feature_idx"); oss << ":" << n.feature_idx << ",";
            w.write_string("split_value"); oss << ":"; w.write_float(n.split_value); oss << ",";
            w.write_string("left_child"); oss << ":" << n.left_child << ",";
            w.write_string("right_child"); oss << ":" << n.right_child << ",";
            w.write_string("leaf_value"); oss << ":"; w.write_float(n.leaf_value); oss << ",";
            w.write_string("num_samples"); oss << ":" << n.num_samples << ",";
            w.write_string("gain"); oss << ":"; w.write_float(n.gain); oss << ",";
            w.write_string("sum_grad"); oss << ":"; w.write_float(n.sum_grad); oss << ",";
            w.write_string("sum_hess"); oss << ":"; w.write_float(n.sum_hess); oss << ",";
            w.write_string("depth"); oss << ":" << n.depth << ",";
            w.write_string("default_left"); oss << ":"; w.write_bool(n.default_left);
            oss << "}";
        }
        oss << "]";
    }
    oss << "],";

    w.write_string("tree_step_sizes");
    oss << ":[";
    for (size_t i = 0; i < tree_step_sizes_.size(); ++i) {
        if (i > 0) oss << ",";
        oss << tree_step_sizes_[i];
    }
    oss << "],";

    w.write_string("metrics");
    oss << ":{";
    w.write_string("num_trees_built"); oss << ":" << metrics_.num_trees_built << ",";
    w.write_string("train_losses"); oss << ":[";
    for (size_t i = 0; i < metrics_.train_losses.size(); ++i) {
        if (i > 0) oss << ",";
        oss << metrics_.train_losses[i];
    }
    oss << "],";
    w.write_string("val_losses"); oss << ":[";
    for (size_t i = 0; i < metrics_.val_losses.size(); ++i) {
        if (i > 0) oss << ",";
        oss << metrics_.val_losses[i];
    }
    oss << "]}";

    oss << "}";
    return oss.str();
}

void GBDT::from_json(const std::string& json_str) {
    JsonParser p(json_str);

    trees_.clear();
    tree_step_sizes_.clear();
    metrics_ = TrainingMetrics{};

    p.expect('{');

    while (p.peek() != '}') {
        std::string key = p.parse_string();
        p.expect(':');

        if (key == "init_pred") {
            init_pred_ = p.parse_number();
        } else if (key == "config") {
            p.expect('{');
            while (p.peek() != '}') {
                std::string ckey = p.parse_string();
                p.expect(':');

                if (ckey == "max_depth") config_.max_depth = p.parse_int();
                else if (ckey == "min_samples_leaf") config_.min_samples_leaf = p.parse_int();
                else if (ckey == "min_gain_to_split") config_.min_gain_to_split = p.parse_number();
                else if (ckey == "max_bins") config_.max_bins = p.parse_int();
                else if (ckey == "lambda_l2") config_.lambda_l2 = p.parse_number();
                else if (ckey == "lambda_l1") config_.lambda_l1 = p.parse_number();
                else if (ckey == "gamma") config_.gamma = p.parse_number();
                else if (ckey == "num_trees") config_.num_trees = p.parse_int();
                else if (ckey == "learning_rate") config_.learning_rate = p.parse_number();
                else if (ckey == "subsample_row") config_.subsample_row = p.parse_number();
                else if (ckey == "subsample_col") config_.subsample_col = p.parse_number();
                else if (ckey == "loss_type") config_.loss_type = p.parse_string();
                else if (ckey == "early_stopping_rounds") config_.early_stopping_rounds = p.parse_int();
                else if (ckey == "early_stopping_tol") config_.early_stopping_tol = p.parse_number();
                else if (ckey == "use_line_search") config_.use_line_search = p.parse_bool();
                else if (ckey == "enable_missing_values") config_.enable_missing_values = p.parse_bool();
                else if (ckey == "huber_delta") config_.huber_delta = p.parse_number();
                else if (ckey == "mae_eps") config_.mae_eps = p.parse_number();

                if (p.peek() == ',') p.consume();
            }
            p.expect('}');
        } else if (key == "trees") {
            p.expect('[');
            while (p.peek() != ']') {
                trees_.push_back(p.parse_tree_nodes());
                if (p.peek() == ',') p.consume();
            }
            p.expect(']');
        } else if (key == "tree_step_sizes") {
            p.expect('[');
            while (p.peek() != ']') {
                tree_step_sizes_.push_back(p.parse_number());
                if (p.peek() == ',') p.consume();
            }
            p.expect(']');
        } else if (key == "metrics") {
            p.expect('{');
            while (p.peek() != '}') {
                std::string mkey = p.parse_string();
                p.expect(':');

                if (mkey == "num_trees_built") {
                    metrics_.num_trees_built = p.parse_int();
                } else if (mkey == "train_losses") {
                    p.expect('[');
                    while (p.peek() != ']') {
                        metrics_.train_losses.push_back(p.parse_number());
                        if (p.peek() == ',') p.consume();
                    }
                    p.expect(']');
                } else if (mkey == "val_losses") {
                    p.expect('[');
                    while (p.peek() != ']') {
                        metrics_.val_losses.push_back(p.parse_number());
                        if (p.peek() == ',') p.consume();
                    }
                    p.expect(']');
                }

                if (p.peek() == ',') p.consume();
            }
            p.expect('}');
        }

        if (p.peek() == ',') p.consume();
    }

    p.expect('}');

    tree_builder_ = std::make_unique<TreeBuilder>(config_);
    rng_.seed(config_.random_seed);
}

void GBDT::set_state(float init_pred,
                     const std::vector<std::vector<TreeNode>>& trees,
                     const std::vector<float>& tree_step_sizes) {
    init_pred_ = init_pred;
    trees_ = trees;
    tree_step_sizes_ = tree_step_sizes;
    metrics_ = TrainingMetrics{};
}

std::vector<float> GBDT::get_feature_importance(int num_features) const {
    std::vector<float> importance(num_features, 0.0f);

    for (const auto& tree : trees_) {
        for (const auto& node : tree) {
            if (!node.is_leaf() &&
                node.feature_idx >= 0 &&
                node.feature_idx < num_features) {
                importance[node.feature_idx] += 1.0f;
            }
        }
    }

    return importance;
}

std::vector<float> GBDT::get_feature_importance_gain(int num_features) const {
    std::vector<float> importance(num_features, 0.0f);

    for (const auto& tree : trees_) {
        for (const auto& node : tree) {
            if (!node.is_leaf() &&
                node.feature_idx >= 0 &&
                node.feature_idx < num_features &&
                node.gain > 0.0f) {
                importance[node.feature_idx] += node.gain;
            }
        }
    }

    float total = 0.0f;
    for (float v : importance) total += v;
    if (total > 0.0f) {
        for (auto& v : importance) v /= total;
    }

    return importance;
}

std::vector<float> GBDT::get_feature_importance_coverage(int num_features) const {
    std::vector<float> importance(num_features, 0.0f);

    for (const auto& tree : trees_) {
        for (const auto& node : tree) {
            if (!node.is_leaf() &&
                node.feature_idx >= 0 &&
                node.feature_idx < num_features) {
                importance[node.feature_idx] += static_cast<float>(node.num_samples);
            }
        }
    }

    float total = 0.0f;
    for (float v : importance) total += v;
    if (total > 0.0f) {
        for (auto& v : importance) v /= total;
    }

    return importance;
}

GBDT::FeatureImportance GBDT::get_feature_importance_full(int num_features) const {
    FeatureImportance imp;
    imp.frequency = std::vector<float>(num_features, 0.0f);
    imp.gain      = std::vector<float>(num_features, 0.0f);
    imp.coverage  = std::vector<float>(num_features, 0.0f);

    for (const auto& tree : trees_) {
        for (const auto& node : tree) {
            if (!node.is_leaf() &&
                node.feature_idx >= 0 &&
                node.feature_idx < num_features) {
                int fidx = node.feature_idx;
                imp.frequency[fidx] += 1.0f;
                imp.coverage[fidx] += static_cast<float>(node.num_samples);
                if (node.gain > 0.0f) {
                    imp.gain[fidx] += node.gain;
                }
            }
        }
    }

    float freq_total = 0.0f;
    float gain_total = 0.0f;
    float cov_total  = 0.0f;
    for (int i = 0; i < num_features; ++i) {
        freq_total += imp.frequency[i];
        gain_total += imp.gain[i];
        cov_total  += imp.coverage[i];
    }

    if (freq_total > 0.0f) {
        for (auto& v : imp.frequency) v /= freq_total;
    }
    if (gain_total > 0.0f) {
        for (auto& v : imp.gain) v /= gain_total;
    }
    if (cov_total > 0.0f) {
        for (auto& v : imp.coverage) v /= cov_total;
    }

    return imp;
}

void GBDT::fit(
    const torch::Tensor& X,
    const torch::Tensor& y,
    const torch::Tensor& X_val,
    const torch::Tensor& y_val,
    const std::vector<MonotoneConstraint>& constraints)
{
    int64_t N = X.size(0);
    if (!X.isfinite().all().item<bool>()) throw std::runtime_error("Non-finite values detected in X at start of fit().");
    if (!y.isfinite().all().item<bool>()) throw std::runtime_error("Non-finite values detected in y at start of fit().");

    bool has_val = X_val.defined() && X_val.numel() > 0 &&
                   y_val.defined() && y_val.numel() > 0;

    trees_.clear();
    tree_step_sizes_.clear();
    metrics_ = TrainingMetrics{};

    // Initial prediction: optimal constant differs by loss type.
    // MSE → mean (minimizes squared error), MAE → median (minimizes absolute error),
    // Huber → mean (close enough to optimal for large delta; median for small delta).
    if (config_.loss_type == "mae") {
        init_pred_ = torch::median(y).item<float>();
    } else {
        init_pred_ = y.mean().item<float>();
    }
    auto y_pred_train = torch::full({N}, init_pred_, torch::kFloat32);

    torch::Tensor y_pred_val;
    if (has_val) {
        y_pred_val = torch::full({X_val.size(0)}, init_pred_, torch::kFloat32);
    }

    float best_val_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    int best_num_trees_snapshot = 0;  // track best model state for rollback

    // Pack loss parameters into tensor: [huber_delta, mae_eps]
    torch::Tensor loss_params = torch::tensor(
        {config_.huber_delta, config_.mae_eps}, torch::kFloat32);

    TORCH_CHECK(loss_params.dtype() == torch::kFloat32, "loss_params must be float32");
    TORCH_CHECK(loss_params.is_cpu(), "loss_params must be on CPU");
    TORCH_CHECK(loss_params.is_contiguous(), "loss_params must be contiguous");

    for (int m = 0; m < config_.num_trees; ++m) {
        auto grad_hess = bridge_->compute_gradients_hessians(
            y, y_pred_train, config_.loss_type, loss_params);
        auto gradients = grad_hess.gradients;
        auto hessians = grad_hess.hessians;

        auto tree = fit_one_tree(X, gradients, hessians, constraints);

        auto tree_pred_full = predict_tree(tree, X);

        float rho = 1.0f;
        if (config_.use_line_search) {
            rho = line_search(gradients, hessians, tree_pred_full);
        }

        float step = config_.learning_rate * rho;
        tree_step_sizes_.push_back(step);
        y_pred_train += step * tree_pred_full;

        if (has_val) {
            auto tree_pred_val = predict_tree(tree, X_val);
            y_pred_val += step * tree_pred_val;
        }

        float train_loss = compute_scalar_loss(y_pred_train, y, config_.loss_type, config_.huber_delta);
        metrics_.train_losses.push_back(train_loss);

        if (has_val) {
            float val_loss = compute_scalar_loss(y_pred_val, y_val, config_.loss_type, config_.huber_delta);
            metrics_.val_losses.push_back(val_loss);

            trees_.push_back(std::move(tree));

            if (val_loss < best_val_loss - config_.early_stopping_tol) {
                best_val_loss = val_loss;
                patience_counter = 0;
                best_num_trees_snapshot = static_cast<int>(trees_.size());
            } else {
                ++patience_counter;
            }

            if (patience_counter >= config_.early_stopping_rounds) {
                // Roll back to the model state that had best validation loss
                if (static_cast<int>(trees_.size()) > best_num_trees_snapshot) {
                    trees_.resize(best_num_trees_snapshot);
                    tree_step_sizes_.resize(best_num_trees_snapshot);
                    metrics_.train_losses.resize(best_num_trees_snapshot);
                    metrics_.val_losses.resize(best_num_trees_snapshot);
                }
                break;
            }
        } else {
            trees_.push_back(std::move(tree));
        }
    }

    metrics_.num_trees_built = static_cast<int>(trees_.size());
}

}  // namespace gbdt
