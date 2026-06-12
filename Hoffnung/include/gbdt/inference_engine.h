#pragma once
#include "tree.h"
#include <torch/torch.h>
#include <vector>
#include <cstdint>
#include <cmath>

namespace gbdt {

// FlatNode: memory-contiguous representation for branchless traversal.
// Unlike TreeNode which has named fields, FlatNode stores data in a
// compact layout for cache-friendly inference.
struct FlatNode {
    float split_value = std::nanf("");   // threshold for split (leaf: NaN)
    int16_t feature_idx = -1;            // split feature (-1 for leaf, int16_t saves memory)
    int16_t left_child = -1;             // left child index
    int16_t right_child = -1;            // right child index
    float leaf_value = 0.0f;             // prediction (leaf only)

    bool is_leaf() const { return feature_idx < 0; }
};

// InferenceEngine: converts tree nodes to flat format for fast batch prediction.
// Uses a per-sample while-loop on FlatNode arrays — the cache-friendly memory layout
// reduces cache misses compared to TreeNode traversal.
class InferenceEngine {
public:
    // Convert a forest (vector of trees) to flat representation.
    static std::vector<std::vector<FlatNode>> convert_forest(
        const std::vector<std::vector<TreeNode>>& trees);

    // Convert a single tree to flat nodes.
    static std::vector<FlatNode> convert_tree(
        const std::vector<TreeNode>& tree);

    // Batch predict using flat trees.
    // X: [N, D] float32 features on CPU
    // flat_trees: forest in flat format
    // init_pred: initial prediction offset
    // tree_weights: per-tree step sizes (lr * rho)
    // Returns: [N] float32 predictions
    static torch::Tensor batch_predict(
        const torch::Tensor& X,
        const std::vector<std::vector<FlatNode>>& flat_trees,
        float init_pred,
        const std::vector<float>& tree_weights);

    // Single-sample predict using flat tree (no batch overhead).
    static float predict_single(
        const float* x_data,
        int64_t num_cols,
        const std::vector<FlatNode>& flat_tree);
};

} // namespace gbdt
