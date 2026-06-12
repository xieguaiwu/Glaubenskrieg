#pragma once
#include <limits>
#include <vector>
#include <string>
#include <torch/torch.h>

namespace gbdt {

struct TreeNode {
    int feature_idx = -1;       // split feature (-1 for leaf)
    float split_value = 0.0f;   // threshold for split
    int left_child = -1;        // index of left child
    int right_child = -1;       // index of right child
    float leaf_value = 0.0f;    // prediction value (leaf only)
    int num_samples = 0;        // samples in this node
    float gain = 0.0f;          // split gain
    float sum_grad = 0.0f;      // sum of gradients in this node
    float sum_hess = 0.0f;      // sum of Hessians in this node
    int depth = 0;              // depth from root
    bool default_left = true;   // direction for missing values

    bool is_leaf() const { return left_child == -1 && right_child == -1; }
};

struct HistogramBin {
    float sum_grad = 0.0f;
    float sum_hess = 0.0f;
    int count = 0;
};

using FeatureHistogram = std::vector<HistogramBin>;

struct Histograms {
    std::vector<FeatureHistogram> feature_histograms;
    int num_bins = 0;
    int num_features = 0;
};

struct SplitResult {
    int feature_idx = -1;
    float threshold = 0.0f;
    float gain = -std::numeric_limits<float>::infinity();
    bool valid = false;
};

// Serialize tree to/from JSON
std::string tree_to_json(const std::vector<TreeNode>& nodes, int num_features);
std::vector<TreeNode> tree_from_json(const std::string& json_str);

}  // namespace gbdt
