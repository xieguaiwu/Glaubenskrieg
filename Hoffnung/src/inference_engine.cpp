#include "gbdt/inference_engine.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gbdt {

std::vector<FlatNode> InferenceEngine::convert_tree(
    const std::vector<TreeNode>& tree)
{
    std::vector<FlatNode> flat;
    flat.reserve(tree.size());

    for (const auto& node : tree) {
        FlatNode fn;

        if (node.is_leaf()) {
            fn.feature_idx = -1;
            fn.split_value = std::nanf("");
            fn.left_child = -1;
            fn.right_child = -1;
            fn.leaf_value = node.leaf_value;
        } else {
            if (node.feature_idx > std::numeric_limits<int16_t>::max() ||
                node.feature_idx < std::numeric_limits<int16_t>::min()) {
                throw std::overflow_error(
                    "feature_idx " + std::to_string(node.feature_idx) +
                    " exceeds int16_t range");
            }
            if (node.left_child > std::numeric_limits<int16_t>::max() ||
                node.left_child < std::numeric_limits<int16_t>::min()) {
                throw std::overflow_error(
                    "left_child " + std::to_string(node.left_child) +
                    " exceeds int16_t range");
            }
            if (node.right_child > std::numeric_limits<int16_t>::max() ||
                node.right_child < std::numeric_limits<int16_t>::min()) {
                throw std::overflow_error(
                    "right_child " + std::to_string(node.right_child) +
                    " exceeds int16_t range");
            }

            fn.feature_idx = static_cast<int16_t>(node.feature_idx);
            fn.split_value = node.split_value;
            fn.left_child = static_cast<int16_t>(node.left_child);
            fn.right_child = static_cast<int16_t>(node.right_child);
            fn.leaf_value = std::nanf("");
        }

        flat.push_back(fn);
    }

    return flat;
}

std::vector<std::vector<FlatNode>> InferenceEngine::convert_forest(
    const std::vector<std::vector<TreeNode>>& trees)
{
    std::vector<std::vector<FlatNode>> flat_forest;
    flat_forest.reserve(trees.size());

    for (const auto& tree : trees) {
        flat_forest.push_back(convert_tree(tree));
    }

    return flat_forest;
}

float InferenceEngine::predict_single(
    const float* x_data,
    int64_t num_cols,
    const std::vector<FlatNode>& flat_tree)
{
    if (flat_tree.empty()) {
        return 0.0f;
    }

    int node = 0;
    const int num_nodes = static_cast<int>(flat_tree.size());

    while (node >= 0 && node < num_nodes) {
        const auto& fn = flat_tree[node];
        if (fn.is_leaf()) {
            return fn.leaf_value;
        }

        int feat = static_cast<int>(fn.feature_idx);
        if (feat < 0 || feat >= num_cols) {
            return 0.0f;
        }

        float val = x_data[feat];

        if (std::isnan(val)) {
            node = static_cast<int>(fn.left_child);
        } else if (val <= fn.split_value) {
            node = static_cast<int>(fn.left_child);
        } else {
            node = static_cast<int>(fn.right_child);
        }
    }

    return 0.0f;
}

torch::Tensor InferenceEngine::batch_predict(
    const torch::Tensor& X,
    const std::vector<std::vector<FlatNode>>& flat_trees,
    float init_pred,
    const std::vector<float>& tree_weights)
{
    int64_t N = X.size(0);
    int64_t D = X.size(1);

    auto y_pred = torch::full({N}, init_pred, torch::kFloat32);
    auto y_pred_acc = y_pred.accessor<float, 1>();

    auto X_contig = X.contiguous();
    const float* x_data = X_contig.const_data_ptr<float>();

    const int num_trees = static_cast<int>(flat_trees.size());

    for (int t = 0; t < num_trees; ++t) {
        const auto& ft = flat_trees[t];
        const int num_nodes = static_cast<int>(ft.size());

        if (num_nodes == 0) continue;

        float step = (t < static_cast<int>(tree_weights.size()))
                         ? tree_weights[t]
                         : 0.1f;

#if defined(_OPENMP)
        #pragma omp parallel for
#endif
        for (int64_t s = 0; s < N; ++s) {
            int node = 0;

            while (node >= 0 && node < num_nodes) {
                const auto& fn = ft[node];
                if (fn.is_leaf()) {
                    y_pred_acc[s] += step * fn.leaf_value;
                    break;
                }

                int feat = static_cast<int>(fn.feature_idx);
                if (feat < 0 || feat >= D) break;

                float val = x_data[s * D + feat];

                if (std::isnan(val)) {
                    node = static_cast<int>(fn.left_child);
                } else if (val <= fn.split_value) {
                    node = static_cast<int>(fn.left_child);
                } else {
                    node = static_cast<int>(fn.right_child);
                }
            }
        }
    }

    return y_pred;
}

} // namespace gbdt
