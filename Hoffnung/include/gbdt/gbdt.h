#pragma once
#include "tree.h"
#include "builder.h"
#include "config.h"
#include "inference_engine.h"
#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace gbdt {

struct GradHessOutput {
    torch::Tensor gradients;
    torch::Tensor hessians;
    float loss_value = 0.0f;
};

float compute_scalar_loss(const torch::Tensor& y_pred, const torch::Tensor& y_true,
                          const std::string& loss_type,
                          float huber_delta = 1.0f);

class GradientBridge {
public:
    GradHessOutput compute_gradients_hessians(
        const torch::Tensor& y_true,
        const torch::Tensor& y_pred,
        const std::string& loss_type,
        const torch::Tensor& loss_params = torch::Tensor()
    );
};

class GBDT {
public:
    explicit GBDT(const GBDTConfig& config);

    void fit(
        const torch::Tensor& X,
        const torch::Tensor& y,
        const torch::Tensor& X_val = torch::Tensor(),
        const torch::Tensor& y_val = torch::Tensor(),
        const std::vector<MonotoneConstraint>& constraints = {}
    );

    torch::Tensor predict(const torch::Tensor& X) const;

    torch::Tensor predict_batch(const torch::Tensor& X) const;

    torch::Tensor predict_tree(const std::vector<TreeNode>& tree,
                               const torch::Tensor& X,
                               const std::vector<int>& col_idx = {}) const;

    std::vector<TreeNode> fit_one_tree(
        const torch::Tensor& X,
        const torch::Tensor& gradients,
        const torch::Tensor& hessians,
        const std::vector<MonotoneConstraint>& constraints = {}
    );

    std::vector<std::vector<TreeNode>> get_trees() const { return trees_; }

    std::string to_json() const;
    void from_json(const std::string& json_str);

    void set_state(float init_pred,
                   const std::vector<std::vector<TreeNode>>& trees,
                   const std::vector<float>& tree_step_sizes);

    std::vector<float> get_feature_importance(int num_features) const;

    // Feature importance — all three types in one struct
    struct FeatureImportance {
        std::vector<float> frequency;  // split count (same as get_feature_importance)
        std::vector<float> gain;       // total gain contributed by splits on each feature
        std::vector<float> coverage;   // total samples routed through split on each feature
    };

    std::vector<float> get_feature_importance_gain(int num_features) const;
    std::vector<float> get_feature_importance_coverage(int num_features) const;
    FeatureImportance get_feature_importance_full(int num_features) const;

    int num_trees() const { return static_cast<int>(trees_.size()); }

    // Training history
    struct TrainingMetrics {
        std::vector<float> train_losses;
        std::vector<float> val_losses;
        int num_trees_built = 0;
    };
    const TrainingMetrics& get_metrics() const { return metrics_; }

private:
    float line_search(
        const torch::Tensor& gradients,
        const torch::Tensor& hessians,
        const torch::Tensor& tree_pred
    ) const;

    std::vector<int> random_subset(int n, float ratio);
    std::vector<int> random_col_subset(int n, float ratio);

    GBDTConfig config_;
    float init_pred_ = 0.0f;
    std::vector<std::vector<TreeNode>> trees_;
    std::vector<float> tree_step_sizes_;  // per-tree step = lr * rho
    std::unique_ptr<TreeBuilder> tree_builder_;
    std::unique_ptr<GradientBridge> bridge_;
    std::mt19937 rng_;
    TrainingMetrics metrics_;
};

}  // namespace gbdt
