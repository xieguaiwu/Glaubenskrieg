#pragma once
#include "tree.h"
#include "config.h"
#include <vector>
#include <memory>
#include <random>

namespace gbdt {

class HistogramBuilder {
public:
    Histograms build_histogram(
        const torch::Tensor& features,
        const torch::Tensor& grads,
        const torch::Tensor& hessians,
        int num_bins,
        const std::vector<std::vector<float>>& bin_boundaries
    );

    std::vector<std::vector<float>> compute_bin_boundaries(
        const torch::Tensor& features,
        int num_bins
    );

};

class SplitFinder {
public:
    SplitResult find_best_split(
        const Histograms& hist,
        float parent_sum_grad,
        float parent_sum_hess,
        float lambda_l2,
        float gamma,
        const std::vector<std::vector<float>>& bin_boundaries,
        const std::vector<MonotoneConstraint>& constraints = {}
    );

private:
    float compute_gain(float left_grad, float left_hess,
                       float right_grad, float right_hess,
                       float parent_grad, float parent_hess,
                       float lambda_l2, float gamma) const;
};

class TreeBuilder {
public:
    TreeBuilder(const GBDTConfig& config);

    std::vector<TreeNode> build_tree(
        const torch::Tensor& features,
        const torch::Tensor& gradients,
        const torch::Tensor& hessians,
        const std::vector<MonotoneConstraint>& constraints = {}
    );

private:
    float compute_leaf_value(float sum_grad, float sum_hess,
                             float lambda_l1, float lambda_l2) const;

    GBDTConfig config_;
    std::unique_ptr<HistogramBuilder> hist_builder_;
    std::unique_ptr<SplitFinder> split_finder_;
    std::mt19937 rng_;
};

}  // namespace gbdt
