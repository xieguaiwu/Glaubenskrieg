#include <gbdt/builder.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <stack>
#include <utility>
#include <vector>

namespace gbdt {

namespace {

int find_bin_impl(float value, const std::vector<float>& boundaries) {
    if (std::isnan(value)) return 0;
    if (boundaries.empty()) return 0;
    auto it = std::upper_bound(boundaries.begin(), boundaries.end(), value);
    return static_cast<int>(it - boundaries.begin());
}

int find_bin_clamped(float value, const std::vector<float>& boundaries, int num_bins) {
    int bin = find_bin_impl(value, boundaries);
    if (bin < 0) return 0;
    if (bin >= num_bins) return num_bins - 1;
    return bin;
}

Histograms build_subset_histogram(
    const std::vector<uint8_t>& bin_indices,
    const float* grads,
    const float* hessians,
    const std::vector<int>& sample_indices,
    int start,
    int end,
    int num_bins,
    int num_features) {

    Histograms result;
    result.num_bins = num_bins;
    result.num_features = num_features;
    result.feature_histograms.resize(num_features,
        FeatureHistogram(num_bins));

    for (int p = start; p < end; ++p) {
        int idx = sample_indices.at(p);
        if (idx < 0) continue;
        float g = grads[idx];
        float h = hessians[idx];
        size_t base = static_cast<size_t>(idx) * num_features;
        for (int f = 0; f < num_features; ++f) {
            int bin = bin_indices[base + f];
            if (bin >= 0 && bin < num_bins) {
                auto& b = result.feature_histograms[f][bin];
                b.sum_grad += g;
                b.sum_hess += h;
                b.count++;
            }
        }
    }

    return result;
}

Histograms build_subset_histogram_parallel(
    const std::vector<uint8_t>& bin_indices,
    const float* grads,
    const float* hessians,
    const std::vector<int>& sample_indices,
    int start,
    int end,
    int num_bins,
    int num_features,
    int num_threads) {

    int actual_threads = std::min(num_threads, omp_get_max_threads());

    std::vector<Histograms> local_hists(actual_threads);
    for (int t = 0; t < actual_threads; ++t) {
        local_hists[t].num_bins = num_bins;
        local_hists[t].num_features = num_features;
        local_hists[t].feature_histograms.resize(num_features,
            FeatureHistogram(num_bins));
    }

    #pragma omp parallel num_threads(actual_threads)
    {
        int tid = omp_get_thread_num();
        auto& local = local_hists[tid];

        #pragma omp for schedule(static)
        for (int p = start; p < end; ++p) {
            int idx = sample_indices.at(p);
            if (idx < 0) continue;
            float g = grads[idx];
            float h = hessians[idx];
            size_t base = static_cast<size_t>(idx) * num_features;
            for (int f = 0; f < num_features; ++f) {
                int bin = bin_indices[base + f];
                if (bin >= 0 && bin < num_bins) {
                    auto& b = local.feature_histograms[f][bin];
                    b.sum_grad += g;
                    b.sum_hess += h;
                    b.count++;
                }
            }
        }
    }

    // Merge thread-local histograms into result
    Histograms result = std::move(local_hists[0]);
    for (int t = 1; t < actual_threads; ++t) {
        for (int f = 0; f < num_features; ++f) {
            for (int b = 0; b < num_bins; ++b) {
                result.feature_histograms[f][b].sum_grad +=
                    local_hists[t].feature_histograms[f][b].sum_grad;
                result.feature_histograms[f][b].sum_hess +=
                    local_hists[t].feature_histograms[f][b].sum_hess;
                result.feature_histograms[f][b].count +=
                    local_hists[t].feature_histograms[f][b].count;
            }
        }
    }

    return result;
}

// Compute right-child histogram by subtracting left-child from parent.
// Histogram accumulation is additive: parent = left + right, so right = parent - left.
// This avoids scanning the right child's samples entirely.
void subtract_histogram(Histograms& target, const Histograms& parent,
                        const Histograms& left) {
    int num_features = parent.num_features;
    int num_bins = parent.num_bins;
    target.num_bins = num_bins;
    target.num_features = num_features;

    if (left.num_features != num_features || left.num_bins != num_bins) {
        target.feature_histograms.clear();
        return;
    }

    target.feature_histograms.resize(num_features, FeatureHistogram(num_bins));

    for (int f = 0; f < num_features; ++f) {
        const auto& parent_feat = parent.feature_histograms[f];
        const auto& left_feat = left.feature_histograms[f];
        auto& target_feat = target.feature_histograms[f];

        for (int b = 0; b < num_bins; ++b) {
            float sg = parent_feat[b].sum_grad - left_feat[b].sum_grad;
            float sh = parent_feat[b].sum_hess - left_feat[b].sum_hess;
            int   ct = parent_feat[b].count - left_feat[b].count;

            // Clamp tiny negative values from floating-point imprecision
            target_feat[b].sum_grad = (sg < 0.0f && sg > -1e-12f) ? 0.0f : sg;
            target_feat[b].sum_hess = (sh < 0.0f && sh > -1e-12f) ? 0.0f : sh;
            target_feat[b].count = (ct < 0) ? 0 : ct;
        }
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// HistogramBuilder
// ---------------------------------------------------------------------------

std::vector<std::vector<float>> HistogramBuilder::compute_bin_boundaries(
    const torch::Tensor& features,
    int num_bins) {

    int64_t F = features.size(1);
    if (num_bins <= 1) {
        return std::vector<std::vector<float>>(F);
    }

    auto features_cpu = features.to(torch::kCPU).contiguous();
    int64_t N = features_cpu.size(0);
    if (N == 0) return std::vector<std::vector<float>>(F);
    auto f_acc = features_cpu.accessor<float, 2>();

    std::vector<std::vector<float>> boundaries(F);

    for (int64_t j = 0; j < F; ++j) {
        auto col_tensor = features_cpu.index({torch::indexing::Slice(), j}).clone();
        auto valid_mask = ~torch::isnan(col_tensor);
        col_tensor = col_tensor.index({valid_mask}).contiguous();
        int64_t valid_N = col_tensor.size(0);

        if (valid_N == 0) {
            boundaries[j].push_back(0.0f);
            continue;
        }

        col_tensor = std::get<0>(torch::sort(col_tensor));
        const float* col_data = col_tensor.data_ptr<float>();

        std::vector<float> unique_vals;
        unique_vals.reserve(valid_N);
        for (int64_t i = 0; i < valid_N; ++i) {
            if (unique_vals.empty() ||
                std::fabs(col_data[i] - unique_vals.back()) >= 1e-12f) {
                unique_vals.push_back(col_data[i]);
            }
        }

        int unique_count = static_cast<int>(unique_vals.size());
        std::vector<float>& feat_bounds = boundaries[j];

        if (unique_count == 1) {
            float val = unique_vals[0];
            float eps = std::max(std::fabs(val) * 1e-5f, 1e-7f);
            for (int k = 0; k < num_bins - 1; ++k) {
                feat_bounds.push_back(val + eps * static_cast<float>(k + 1));
            }
        } else if (unique_count < num_bins) {
            for (int k = 0; k < unique_count - 1; ++k) {
                feat_bounds.push_back(
                    (unique_vals[k] + unique_vals[k + 1]) / 2.0f);
            }
        } else {
            for (int k = 1; k < num_bins; ++k) {
                double pct = static_cast<double>(k) / num_bins;
                auto pick = static_cast<int64_t>(pct * (unique_count - 1));
                feat_bounds.push_back(unique_vals[pick]);
            }
            std::vector<float> clean;
            for (float b : feat_bounds) {
                if (clean.empty() || b > clean.back() + 1e-10f) {
                    clean.push_back(b);
                }
            }
            feat_bounds = std::move(clean);
        }
    }

    return boundaries;
}

Histograms HistogramBuilder::build_histogram(
    const torch::Tensor& features,
    const torch::Tensor& grads,
    const torch::Tensor& hessians,
    int num_bins,
    const std::vector<std::vector<float>>& bin_boundaries) {

    auto features_cpu = features.to(torch::kCPU).contiguous();
    auto grads_cpu = grads.to(torch::kCPU).contiguous();
    auto hess_cpu = hessians.to(torch::kCPU).contiguous();

    int64_t N = features_cpu.size(0);
    int64_t F = features_cpu.size(1);

    // Discretize features into bin indices using the provided boundaries
    auto f_acc = features_cpu.accessor<float, 2>();
    std::vector<uint8_t> bin_indices(static_cast<size_t>(N) * F);
    for (int64_t i = 0; i < N; ++i) {
        size_t base = static_cast<size_t>(i) * F;
        for (int64_t j = 0; j < F; ++j) {
            int bin = find_bin_clamped(f_acc[i][j], bin_boundaries[j], num_bins);
            bin_indices[base + j] = static_cast<uint8_t>(bin);
        }
    }

    auto g_acc = grads_cpu.accessor<float, 1>();
    auto h_acc = hess_cpu.accessor<float, 1>();

    std::vector<int> all_indices(N);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    return build_subset_histogram(
        bin_indices, g_acc.data(), h_acc.data(), all_indices,
        0, static_cast<int>(N), num_bins, static_cast<int>(F));
}

// ---------------------------------------------------------------------------
// SplitFinder
// ---------------------------------------------------------------------------

float SplitFinder::compute_gain(
    float left_grad, float left_hess,
    float right_grad, float right_hess,
    float parent_grad, float parent_hess,
    float lambda_l2, float gamma) const {

    auto safe_div = [](float g, float h, float lambda) -> float {
        float denom = h + lambda;
        if (denom <= std::numeric_limits<float>::min()) return 0.0f;
        return (g * g) / denom;
    };

    float left_gain = safe_div(left_grad, left_hess, lambda_l2);
    float right_gain = safe_div(right_grad, right_hess, lambda_l2);
    float parent_gain = safe_div(parent_grad, parent_hess, lambda_l2);

    return left_gain + right_gain - parent_gain - gamma;
}

SplitResult SplitFinder::find_best_split(
    const Histograms& hist,
    float parent_sum_grad,
    float parent_sum_hess,
    float lambda_l2,
    float gamma,
    const std::vector<std::vector<float>>& bin_boundaries,
    const std::vector<MonotoneConstraint>& constraints) {

    SplitResult best;

    for (int f = 0; f < hist.num_features; ++f) {
        if (f >= static_cast<int>(hist.feature_histograms.size())) break;
        const auto& feat_hist = hist.feature_histograms[f];
        int actual_bins = std::min(hist.num_bins, static_cast<int>(feat_hist.size()));

        float left_grad = 0.0f;
        float left_hess = 0.0f;

        for (int bin_idx = 0; bin_idx < actual_bins - 1; ++bin_idx) {
            left_grad += feat_hist[bin_idx].sum_grad;
            left_hess += feat_hist[bin_idx].sum_hess;

            float right_grad = parent_sum_grad - left_grad;
            float right_hess = parent_sum_hess - left_hess;

            if (left_hess <= 0.0f || right_hess <= 0.0f) continue;

            float gain = compute_gain(
                left_grad, left_hess,
                right_grad, right_hess,
                parent_sum_grad, parent_sum_hess,
                lambda_l2, gamma);

            if (!constraints.empty()) {
                bool skip = false;
                for (const auto& c : constraints) {
                    if (c.feature_idx != f || c.dir == MonotoneConstraint::NONE) {
                        continue;
                    }
                    float denom_l = left_hess + lambda_l2;
                    float denom_r = right_hess + lambda_l2;
                    float left_val = (denom_l > 1e-20f) ? -left_grad / denom_l : 0.0f;
                    float right_val = (denom_r > 1e-20f) ? -right_grad / denom_r : 0.0f;
                    if (c.dir == MonotoneConstraint::INCREASING && left_val > right_val) {
                        skip = true; break;
                    }
                    if (c.dir == MonotoneConstraint::DECREASING && left_val < right_val) {
                        skip = true; break;
                    }
                }
                if (skip) continue;
            }

            if (gain > best.gain) {
                best.gain = gain;
                best.feature_idx = f;
                if (f < static_cast<int>(bin_boundaries.size())) {
                    const auto& bounds = bin_boundaries[f];
                    if (bin_idx < static_cast<int>(bounds.size())) {
                        best.threshold = bounds[bin_idx];
                    }
                }
                best.valid = true;
            }
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// TreeBuilder
// ---------------------------------------------------------------------------

TreeBuilder::TreeBuilder(const GBDTConfig& config)
    : config_(config)
    , hist_builder_(std::make_unique<HistogramBuilder>())
    , split_finder_(std::make_unique<SplitFinder>())
    , rng_(config.random_seed)
{
}

float TreeBuilder::compute_leaf_value(
    float sum_grad, float sum_hess,
    float lambda_l1, float lambda_l2) const {

    // XGBoost-compatible L1/L2 regularization:
    //   w = -sign(sum_grad) * max(|sum_grad| - lambda_l1, 0) / (sum_hess + lambda_l2)
    //
    // L1 is applied to the gradient sum BEFORE dividing by Hessian (standard GBDT
    // practice). This ensures soft shrinkage at the gradient level, consistent with
    // XGBoost and LightGBM.

    float denom = sum_hess + lambda_l2;
    if (denom <= 1e-20f) return 0.0f;

    float shrunk_grad = sum_grad;
    if (lambda_l1 > 0.0f) {
        float abs_grad = std::fabs(shrunk_grad);
        if (abs_grad <= lambda_l1) {
            return 0.0f;  // L1 threshold completely zeroes out leaf
        }
        shrunk_grad = (shrunk_grad > 0.0f)
                          ? (shrunk_grad - lambda_l1)
                          : (shrunk_grad + lambda_l1);
    }

    return -shrunk_grad / denom;
}

std::vector<TreeNode> TreeBuilder::build_tree(
    const torch::Tensor& features,
    const torch::Tensor& gradients,
    const torch::Tensor& hessians,
    const std::vector<MonotoneConstraint>& constraints) {

    auto features_cpu = features.to(torch::kCPU).contiguous();
    auto grads_cpu = gradients.to(torch::kCPU).contiguous();
    auto hess_cpu = hessians.to(torch::kCPU).contiguous();

    int64_t N = features_cpu.size(0);
    int64_t F = features_cpu.size(1);
    if (N == 0 || F == 0) return {};

    int num_bins = config_.max_bins;
    if (num_bins < 2) num_bins = 2;
    if (num_bins > 256) {
        throw std::invalid_argument("max_bins exceeds 256 (uint8_t limit)");
    }

    auto bin_boundaries = hist_builder_->compute_bin_boundaries(features, num_bins);

    auto f_acc = features_cpu.accessor<float, 2>();
    std::vector<uint8_t> bin_indices(static_cast<size_t>(N) * F);
    for (int64_t i = 0; i < N; ++i) {
        size_t base = static_cast<size_t>(i) * F;
        for (int64_t j = 0; j < F; ++j) {
            int bin = find_bin_clamped(f_acc[i][j], bin_boundaries[j], num_bins);
            bin_indices[base + j] = static_cast<uint8_t>(bin);
        }
    }

    std::vector<int> sample_indices(static_cast<size_t>(N));
    std::iota(sample_indices.begin(), sample_indices.end(), 0);

    auto g_acc = grads_cpu.accessor<float, 1>();
    auto h_acc = hess_cpu.accessor<float, 1>();
    const float* g_ptr = g_acc.data();
    const float* h_ptr = h_acc.data();
    if (!g_ptr || !h_ptr) return {};

    float lambda_l2 = config_.lambda_l2;
    float lambda_l1 = config_.lambda_l1;
    float gamma = config_.gamma;
    float min_gain = config_.min_gain_to_split;
    int max_depth = config_.max_depth;
    int min_leaf = config_.min_samples_leaf;

    std::vector<TreeNode> nodes;
    size_t max_nodes = static_cast<size_t>(1u) << (max_depth + 1);
    if (max_nodes > (1u << 20)) max_nodes = 1u << 20;
    nodes.reserve(max_nodes);
    std::vector<Histograms> node_histograms;
    node_histograms.reserve(max_nodes);

    struct BuildTask {
        int start; int end; int depth; int node_idx;
        int parent_node_idx;  // -1 for root
        bool is_right_child;
    };
    std::stack<BuildTask> tasks;

    nodes.push_back(TreeNode{});
    tasks.push({0, static_cast<int>(N), 0, 0, -1, false});

    while (!tasks.empty()) {
        BuildTask task = tasks.top();
        tasks.pop();

        int start = task.start;
        int end = task.end;
        int depth = task.depth;
        int node_idx = task.node_idx;
        int num_samples = end - start;

        if (node_idx < 0 || static_cast<size_t>(node_idx) >= nodes.size()) continue;
        TreeNode& node = nodes[node_idx];
        node.depth = depth;
        node.num_samples = num_samples;

        float node_sum_grad = 0.0f;
        float node_sum_hess = 0.0f;
        if (start < 0) start = 0;
        if (end > static_cast<int>(N)) end = static_cast<int>(N);
        for (int p = start; p < end; ++p) {
            int idx = p < static_cast<int>(sample_indices.size()) ? sample_indices[p] : -1;
            if (idx < 0 || idx >= N) continue;
            node_sum_grad += g_ptr[idx];
            node_sum_hess += h_ptr[idx];
        }
        node.sum_grad = node_sum_grad;
        node.sum_hess = node_sum_hess;

        if (depth >= max_depth || num_samples < 2) {
            node.feature_idx = -1;
            node.leaf_value = compute_leaf_value(
                node_sum_grad, node_sum_hess, lambda_l1, lambda_l2);
            continue;
        }

        // Use parallel histogram for large node subsets (threshold = 4096)
        constexpr int PARALLEL_HIST_THRESHOLD = 4096;

        // Ensure node_histograms has capacity for this node
        if (static_cast<int>(node_histograms.size()) <= node_idx) {
            node_histograms.resize(node_idx + 1);
        }

        bool built_by_subtraction = false;

        // If this is a right child with enough left-sibling samples,
        // compute histogram by subtracting left from parent (no scan needed).
        if (task.is_right_child && task.parent_node_idx >= 0) {
            int parent_idx = task.parent_node_idx;
            if (parent_idx < 0 || static_cast<size_t>(parent_idx) >= nodes.size()) continue;
            int left_sibling_idx = nodes[parent_idx].left_child;
            if (left_sibling_idx < 0 || static_cast<size_t>(left_sibling_idx) >= nodes.size()) continue;
            int left_samples = nodes[left_sibling_idx].num_samples;
            if (left_samples > 10 &&
                static_cast<size_t>(parent_idx) < node_histograms.size() &&
                static_cast<size_t>(left_sibling_idx) < node_histograms.size()) {
                subtract_histogram(node_histograms[node_idx],
                                   node_histograms[parent_idx],
                                   node_histograms[left_sibling_idx]);
                built_by_subtraction = true;
            }
        }

        if (!built_by_subtraction) {
            node_histograms[node_idx] = (num_samples > PARALLEL_HIST_THRESHOLD && config_.num_threads > 1)
                ? build_subset_histogram_parallel(
                    bin_indices, g_ptr, h_ptr, sample_indices,
                    start, end, num_bins, static_cast<int>(F), config_.num_threads)
                : build_subset_histogram(
                    bin_indices, g_ptr, h_ptr, sample_indices,
                    start, end, num_bins, static_cast<int>(F));
        }

        const Histograms& hist = node_histograms[node_idx];

        SplitResult split = split_finder_->find_best_split(
            hist, node_sum_grad, node_sum_hess,
            lambda_l2, gamma, bin_boundaries, constraints);

        node.gain = split.gain;

        bool accept_split = split.valid &&
            split.gain > min_gain &&
            num_samples >= 2 * min_leaf;

        if (!accept_split) {
            node.feature_idx = -1;
            node.leaf_value = compute_leaf_value(
                node_sum_grad, node_sum_hess, lambda_l1, lambda_l2);
            continue;
        }

        node.feature_idx = split.feature_idx;
        node.split_value = split.threshold;

        int split_f = split.feature_idx;
        if (split_f < 0 || split_f >= static_cast<int>(bin_boundaries.size())) {
            node.feature_idx = -1;
            node.leaf_value = compute_leaf_value(
                node_sum_grad, node_sum_hess, lambda_l1, lambda_l2);
            continue;
        }
        int split_bin = find_bin_clamped(split.threshold, bin_boundaries[split_f], num_bins);

        int left = start;
        int right = end - 1;
        if (left < 0) left = 0;
        if (right >= static_cast<int>(sample_indices.size())) right = static_cast<int>(sample_indices.size()) - 1;
        if (end > static_cast<int>(sample_indices.size())) end = static_cast<int>(sample_indices.size());
        while (left <= right) {
            // Find left element that belongs on the right side
            while (left <= right && left < static_cast<int>(sample_indices.size())) {
                int idx = sample_indices[left];
                int bin = static_cast<int>(bin_indices[static_cast<size_t>(idx) * F + split_f]);
                bool goes_left;
                if (config_.enable_missing_values && std::isnan(f_acc[idx][split_f])) {
                    goes_left = node.default_left;
                } else {
                    goes_left = (bin <= split_bin);
                }
                if (!goes_left) break;
                ++left;
            }
            // Find right element that belongs on the left side
            while (left <= right && right >= 0) {
                int idx = sample_indices[right];
                int bin = static_cast<int>(bin_indices[static_cast<size_t>(idx) * F + split_f]);
                bool goes_left;
                if (config_.enable_missing_values && std::isnan(f_acc[idx][split_f])) {
                    goes_left = node.default_left;
                } else {
                    goes_left = (bin <= split_bin);
                }
                if (goes_left) break;
                --right;
            }
            if (left < right) {
                std::swap(sample_indices[left], sample_indices[right]);
                ++left;
                --right;
            }
        }

        int mid = left;
        int left_count = mid - start;
        int right_count = end - mid;

        if (left_count < min_leaf || right_count < min_leaf) {
            node.feature_idx = -1;
            node.leaf_value = compute_leaf_value(
                node_sum_grad, node_sum_hess, lambda_l1, lambda_l2);
            continue;
        }

        int left_idx = static_cast<int>(nodes.size());
        int right_idx = left_idx + 1;
        node.left_child = left_idx;
        node.right_child = right_idx;
        node.default_left = config_.enable_missing_values;

        nodes.push_back(TreeNode{});
        nodes.push_back(TreeNode{});

        tasks.push({mid, end, depth + 1, right_idx, node_idx, true});
        tasks.push({start, mid, depth + 1, left_idx, node_idx, false});
    }

    // Release histogram memory before returning (subtraction dependencies
    // prevent eager freeing during construction in the iterative builder).
    for (auto& h : node_histograms) {
        h.feature_histograms.clear();
        h.feature_histograms.shrink_to_fit();
    }

    return nodes;
}

}  // namespace gbdt
