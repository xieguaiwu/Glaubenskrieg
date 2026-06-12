#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <limits>
#include <cstdlib>
#include <torch/torch.h>

#include <gbdt/tree.h>
#include <gbdt/builder.h>
#include <gbdt/config.h>
#include <gbdt/gbdt.h>
#include <gbdt/inference_engine.h>

namespace gbdt_test {

bool approx_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

int test_assert(bool condition, const char* name) {
    if (condition) {
        std::cout << "  [PASS] " << name << "\n";
        return 0;
    } else {
        std::cout << "  [FAIL] " << name << "\n";
        return 1;
    }
}

int test_assert_approx(float a, float b, const char* name, float eps = 1e-4f) {
    if (approx_equal(a, b, eps)) {
        std::cout << "  [PASS] " << name << "\n";
        return 0;
    } else {
        std::cout << "  [FAIL] " << name << "  (" << a << " vs " << b << ")\n";
        return 1;
    }
}

int test_treenode_structure() {
    int failures = 0;
    std::cout << "Test 1: TreeNode structure\n";

    gbdt::TreeNode node;
    failures += test_assert(node.is_leaf(),               "default-constructed is leaf");
    failures += test_assert(node.feature_idx == -1,       "feature_idx initialized to -1");
    failures += test_assert(node.split_value == 0.0f,     "split_value initialized to 0.0");
    failures += test_assert(node.left_child == -1,        "left_child initialized to -1");
    failures += test_assert(node.right_child == -1,       "right_child initialized to -1");
    failures += test_assert(node.leaf_value == 0.0f,      "leaf_value initialized to 0.0");
    failures += test_assert(node.num_samples == 0,        "num_samples initialized to 0");
    failures += test_assert(node.gain == 0.0f,            "gain initialized to 0.0");
    failures += test_assert(node.sum_grad == 0.0f,        "sum_grad initialized to 0.0");
    failures += test_assert(node.sum_hess == 0.0f,        "sum_hess initialized to 0.0");
    failures += test_assert(node.depth == 0,              "depth initialized to 0");
    failures += test_assert(node.default_left == true,    "default_left initialized to true");

    node.left_child = 1;
    node.right_child = 2;
    failures += test_assert(!node.is_leaf(),              "not leaf after setting children");
    failures += test_assert(node.left_child == 1,         "left_child == 1");
    failures += test_assert(node.right_child == 2,        "right_child == 2");

    node.feature_idx = 3;
    node.split_value = 0.75f;
    failures += test_assert(node.feature_idx == 3,        "feature_idx updated to 3");
    failures += test_assert_approx(node.split_value, 0.75f, "split_value updated to 0.75");

    node.left_child = -1;
    node.right_child = -1;
    failures += test_assert(node.is_leaf(),               "reverted to leaf after clearing children");

    if (failures == 0)
        std::cout << "  => ALL PASS (0 failures)\n";
    return failures;
}

int test_histogram_builder() {
    int failures = 0;
    std::cout << "Test 2: HistogramBuilder\n";

    auto features = torch::tensor({
        {0.1f, 0.5f},
        {0.3f, 0.7f},
        {0.6f, 0.2f},
        {0.9f, 0.8f}
    });
    auto grads    = torch::tensor({-0.5f, -0.3f, 0.2f, 0.6f});
    auto hessians = torch::tensor({1.0f, 1.0f, 1.0f, 1.0f});

    const int num_bins = 4;
    gbdt::HistogramBuilder builder;

    auto boundaries = builder.compute_bin_boundaries(features, num_bins);
    failures += test_assert(boundaries.size() == 2,                       "boundaries for 2 features");
    for (size_t f = 0; f < boundaries.size(); ++f) {
        bool ok = boundaries[f].size() == static_cast<size_t>(num_bins - 1);
        std::string msg = "boundaries[" + std::to_string(f) + "] has " + std::to_string(num_bins - 1) + " values";
        failures += test_assert(ok, msg.c_str());
    }

    auto hist = builder.build_histogram(features, grads, hessians, num_bins, boundaries);
    failures += test_assert(hist.num_features == 2,                       "hist.num_features == 2");
    failures += test_assert(hist.num_bins    == 4,                        "hist.num_bins == 4");
    failures += test_assert(hist.feature_histograms.size() == 2,          "feature_histograms size == 2");

    for (int f = 0; f < hist.num_features; ++f) {
        auto& fh = hist.feature_histograms[f];
        std::string size_msg = "feature " + std::to_string(f) + " has " + std::to_string(num_bins) + " bins";
        failures += test_assert(fh.size() == static_cast<size_t>(num_bins), size_msg.c_str());

        int feature_count = 0;
        for (int b = 0; b < num_bins; ++b)
            feature_count += fh[b].count;
        std::string cnt_msg = "feature " + std::to_string(f) + " count sum == 4";
        failures += test_assert(feature_count == 4, cnt_msg.c_str());
    }

    int total_count = 0;
    for (int f = 0; f < hist.num_features; ++f)
        for (int b = 0; b < num_bins; ++b)
            total_count += hist.feature_histograms[f][b].count;
    failures += test_assert(total_count == 8, "total count == samples * features (= 8)");

    if (failures == 0)
        std::cout << "  => ALL PASS (0 failures)\n";
    return failures;
}

int test_split_finder() {
    int failures = 0;
    std::cout << "Test 3: SplitFinder\n";

    auto features = torch::tensor({
        {0.1f},
        {0.2f},
        {0.8f},
        {0.9f}
    });
    auto grads    = torch::tensor({-1.0f, -1.0f, 1.0f, 1.0f});
    auto hessians = torch::tensor({1.0f, 1.0f, 1.0f, 1.0f});

    const int num_bins = 4;
    gbdt::HistogramBuilder hist_builder;
    auto boundaries = hist_builder.compute_bin_boundaries(features, num_bins);
    auto hist = hist_builder.build_histogram(features, grads, hessians, num_bins, boundaries);

    float parent_sum_grad = grads.sum().item<float>();
    float parent_sum_hess = hessians.sum().item<float>();

    gbdt::SplitFinder finder;
    auto result = finder.find_best_split(hist, parent_sum_grad, parent_sum_hess,
                                         1.0f, 0.0f, boundaries);

    failures += test_assert(result.valid,                          "split is valid");
    failures += test_assert(result.feature_idx == 0,               "best split on feature 0");
    failures += test_assert(result.gain > 0.0f,                    "gain is positive");
    failures += test_assert(result.gain < 100.0f,                  "gain is finite (not huge)");
    failures += test_assert(result.threshold >= 0.0f,              "threshold is non-negative");

    if (failures == 0)
        std::cout << "  => ALL PASS (0 failures)\n";
    return failures;
}

int test_tree_builder() {
    int failures = 0;
    std::cout << "Test 4: TreeBuilder\n";

    auto X = torch::randn({100, 3});
    auto y = X.index({"...", 0}) + 0.5f * X.index({"...", 1});
    auto grads    = -y;
    auto hessians = torch::ones_like(y);

    gbdt::GBDTConfig config;
    config.max_depth        = 3;
    config.min_samples_leaf = 5;

    gbdt::TreeBuilder builder(config);
    auto tree = builder.build_tree(X, grads, hessians);

    failures += test_assert(!tree.empty(),                         "tree is not empty");

    int internal_nodes = 0;
    int leaf_nodes     = 0;
    for (const auto& node : tree) {
        if (node.is_leaf()) {
            ++leaf_nodes;
            bool finite = std::isfinite(node.leaf_value);
            failures += test_assert(finite, "leaf value is finite");
        } else {
            ++internal_nodes;
            bool valid_feature = node.feature_idx >= 0 && node.feature_idx < 3;
            failures += test_assert(valid_feature, "internal node has valid feature_idx");
        }
    }

    failures += test_assert(internal_nodes > 0,                    "at least one internal node exists");
    failures += test_assert(leaf_nodes > 0,                        "at least one leaf node exists");

    failures += test_assert(tree.size() == static_cast<size_t>(internal_nodes + leaf_nodes),
                            "tree size matches internal + leaf count");
    for (const auto& node : tree) {
        if (!node.is_leaf()) {
            bool left_ok  = node.left_child  >= 0 && static_cast<size_t>(node.left_child)  < tree.size();
            bool right_ok = node.right_child >= 0 && static_cast<size_t>(node.right_child) < tree.size();
            failures += test_assert(left_ok,  "left_child index in bounds");
            failures += test_assert(right_ok, "right_child index in bounds");
        }
    }

    if (failures == 0)
        std::cout << "  => ALL PASS (0 failures)\n";
    return failures;
}

int test_gbdt_training() {
    int failures = 0;
    std::cout << "Test 5: GBDT training\n";

    const int n_samples  = 500;
    const int n_features = 5;
    const int n_train    = 400;
    const int n_val      = n_samples - n_train;

    auto X = torch::randn({n_samples, n_features});
    auto y = X.index({"...", 0})
           + 0.5f * X.index({"...", 1})
           + 0.25f * X.index({"...", 2})
           + 0.1f * torch::randn({n_samples});

    auto X_train = X.index({torch::indexing::Slice(0, n_train)});
    auto y_train = y.index({torch::indexing::Slice(0, n_train)});
    auto X_val   = X.index({torch::indexing::Slice(n_train, n_samples)});
    auto y_val   = y.index({torch::indexing::Slice(n_train, n_samples)});

    gbdt::GBDTConfig config;
    config.num_trees        = 10;
    config.max_depth        = 4;
    config.min_samples_leaf = 5;
    config.learning_rate    = 0.1f;

    gbdt::GBDT model(config);
    model.fit(X_train, y_train, X_val, y_val);

    failures += test_assert(model.num_trees() == 10, "num_trees == 10");

    const auto& metrics = model.get_metrics();
    failures += test_assert(!metrics.train_losses.empty(), "train_losses is not empty");
    failures += test_assert(metrics.train_losses.size() == 10, "train_losses length == num_trees");

    float first_loss = metrics.train_losses.front();
    float last_loss  = metrics.train_losses.back();
    failures += test_assert(first_loss > last_loss, "train loss decreased (first > last)");

    if (!metrics.val_losses.empty()) {
        failures += test_assert(metrics.val_losses.front() >= metrics.val_losses.back() - 1e-2f,
                                "val loss did not explode");
    }

    auto preds = model.predict(X_train);
    failures += test_assert(preds.sizes().size() == 1, "prediction is 1-D");
    failures += test_assert(preds.size(0) == n_train,  "prediction length == n_train");

    bool all_finite = true;
    for (int i = 0; i < n_train; ++i) {
        if (!std::isfinite(preds[i].item<float>())) {
            all_finite = false;
            break;
        }
    }
    failures += test_assert(all_finite, "all predictions are finite");

    if (failures == 0)
        std::cout << "  => ALL PASS (0 failures)\n";
    return failures;
}

int test_json_serialization() {
    int failures = 0;
    std::cout << "Test 6: JSON serialization\n";

    // Build a small non-trivial tree manually.
    //      [0] feature 0 <= 0.5
    //      /                       \
    //   [1] feature 1 <= 0.3     [2] leaf (value=0.5)
    //   /          \
    // [3] leaf    [4] leaf
    // (-0.3)      (0.2)
    //
    std::vector<gbdt::TreeNode> original(5);

    original[0].feature_idx  = 0;
    original[0].split_value  = 0.5f;
    original[0].left_child   = 1;
    original[0].right_child  = 2;
    original[0].num_samples  = 100;
    original[0].gain         = 10.5f;
    original[0].sum_grad     = -5.0f;
    original[0].sum_hess     = 100.0f;
    original[0].depth        = 0;
    original[0].default_left = true;

    original[1].feature_idx  = 1;
    original[1].split_value  = 0.3f;
    original[1].left_child   = 3;
    original[1].right_child  = 4;
    original[1].num_samples  = 60;
    original[1].gain         = 4.2f;
    original[1].sum_grad     = -3.0f;
    original[1].sum_hess     = 60.0f;
    original[1].depth        = 1;
    original[1].default_left = false;

    original[2].leaf_value   = 0.5f;
    original[2].num_samples  = 40;
    original[2].sum_grad     = -2.0f;
    original[2].sum_hess     = 40.0f;
    original[2].depth        = 1;

    original[3].leaf_value   = -0.3f;
    original[3].num_samples  = 25;
    original[3].sum_grad     = -1.0f;
    original[3].sum_hess     = 25.0f;
    original[3].depth        = 2;

    original[4].leaf_value   = 0.2f;
    original[4].num_samples  = 35;
    original[4].sum_grad     = -2.0f;
    original[4].sum_hess     = 35.0f;
    original[4].depth        = 2;

    std::string json = gbdt::tree_to_json(original, 2);
    failures += test_assert(!json.empty(),                           "JSON output not empty");

    failures += test_assert(json.find("feature_idx") != std::string::npos, "JSON contains feature_idx");
    failures += test_assert(json.find("split_value") != std::string::npos, "JSON contains split_value");
    failures += test_assert(json.find("leaf_value")  != std::string::npos, "JSON contains leaf_value");
    failures += test_assert(json.find("is_leaf")     != std::string::npos, "JSON contains is_leaf");

    auto restored = gbdt::tree_from_json(json);
    failures += test_assert(restored.size() == original.size(),      "same number of nodes after round-trip");
    if (restored.size() == original.size()) {
        for (size_t i = 0; i < original.size(); ++i) {
            std::string prefix = "node[" + std::to_string(i) + "]";

            failures += test_assert(restored[i].feature_idx  == original[i].feature_idx,
                                    (prefix + " feature_idx").c_str());
            failures += test_assert_approx(restored[i].split_value, original[i].split_value,
                                           (prefix + " split_value").c_str());
            failures += test_assert(restored[i].leaf_value   == original[i].leaf_value,
                                    (prefix + " leaf_value").c_str());
            failures += test_assert(restored[i].left_child   == original[i].left_child,
                                    (prefix + " left_child").c_str());
            failures += test_assert(restored[i].right_child  == original[i].right_child,
                                    (prefix + " right_child").c_str());
            failures += test_assert(restored[i].depth        == original[i].depth,
                                    (prefix + " depth").c_str());
            failures += test_assert(restored[i].num_samples  == original[i].num_samples,
                                    (prefix + " num_samples").c_str());
            failures += test_assert_approx(restored[i].gain,     original[i].gain,
                                           (prefix + " gain").c_str());
            failures += test_assert_approx(restored[i].sum_grad, original[i].sum_grad,
                                           (prefix + " sum_grad").c_str());
            failures += test_assert_approx(restored[i].sum_hess, original[i].sum_hess,
                                           (prefix + " sum_hess").c_str());
            failures += test_assert(restored[i].default_left == original[i].default_left,
                                    (prefix + " default_left").c_str());
            failures += test_assert(restored[i].is_leaf()     == original[i].is_leaf(),
                                    (prefix + " is_leaf").c_str());
        }
    }

    std::vector<gbdt::TreeNode> empty;
    std::string empty_json = gbdt::tree_to_json(empty, 0);
    auto empty_restored = gbdt::tree_from_json(empty_json);
    failures += test_assert(empty_restored.empty(), "empty tree round-trips to empty");

    auto from_empty_str = gbdt::tree_from_json("");
    failures += test_assert(from_empty_str.empty(), "tree_from_json(\"\") returns empty");

    if (failures == 0)
        std::cout << "  => ALL PASS (0 failures)\n";
    return failures;
}

int test_feature_importance() {
    int failures = 0;
    std::cout << "Test 7: Feature importance\n";

    const int n_samples  = 500;
    const int n_features = 5;

    auto X = torch::randn({n_samples, n_features});
    auto y = X.index({"...", 0}) + 0.5f * X.index({"...", 1})
           + 0.25f * X.index({"...", 2}) + 0.1f * torch::randn({n_samples});

    gbdt::GBDTConfig config;
    config.num_trees        = 10;
    config.max_depth        = 4;
    config.learning_rate    = 0.1f;
    config.min_samples_leaf = 5;

    gbdt::GBDT model(config);
    model.fit(X, y);

    auto importance = model.get_feature_importance(n_features);

    failures += test_assert(importance.size() == static_cast<size_t>(n_features),
                            "importance vector size == num_features (= 5)");

    int total_importance = 0;
    for (size_t i = 0; i < importance.size(); ++i) {
        failures += test_assert(importance[i] >= 0.0f,
                                ("importance[" + std::to_string(i) + "] >= 0").c_str());
        total_importance += static_cast<int>(importance[i]);
    }

    failures += test_assert(total_importance > 0, "total feature importance > 0");

    if (failures == 0)
        std::cout << "  => ALL PASS (0 failures)\n";
    return failures;
}

int test_column_subsampling() {
    int failures = 0;
    std::cout << "Test 8: Column subsampling\n";

    const int n_samples  = 500;
    const int n_features = 10;

    auto X = torch::randn({n_samples, n_features});
    auto y = X.index({"...", 0}) + X.index({"...", 1}) + 0.1f * torch::randn({n_samples});

    gbdt::GBDTConfig config;
    config.num_trees        = 10;
    config.max_depth        = 4;
    config.learning_rate    = 0.1f;
    config.min_samples_leaf = 5;
    config.subsample_col    = 0.5f;

    gbdt::GBDT model(config);
    model.fit(X, y);

    failures += test_assert(model.num_trees() == 10,
                            "num_trees == 10 after column subsampling");

    auto preds = model.predict(X);
    failures += test_assert(preds.size(0) == n_samples,
                            "prediction shape matches n_samples");
    failures += test_assert(preds.sizes().size() == 1,
                            "prediction is 1-D");

    const auto& metrics = model.get_metrics();
    failures += test_assert(!metrics.train_losses.empty(),
                            "train_losses available after col subsampling");

    bool all_finite = true;
    for (float loss : metrics.train_losses) {
        if (!std::isfinite(loss)) { all_finite = false; break; }
    }
    failures += test_assert(all_finite, "all train losses are finite");

    if (failures == 0)
        std::cout << "  => ALL PASS (0 failures)\n";
    return failures;
}

int test_feature_importance_extended() {
    int failures = 0;
    std::cout << "Test 9: Feature importance — gain & coverage\n";

    const int n_samples  = 500;
    const int n_features = 5;

    auto X = torch::randn({n_samples, n_features});
    auto y = X.index({"...", 0}) + 0.5f * X.index({"...", 1})
           + 0.25f * X.index({"...", 2}) + 0.1f * torch::randn({n_samples});

    gbdt::GBDTConfig config;
    config.num_trees        = 10;
    config.max_depth        = 4;
    config.learning_rate    = 0.1f;
    config.min_samples_leaf = 5;

    gbdt::GBDT model(config);
    model.fit(X, y);

    auto imp_gain     = model.get_feature_importance_gain(n_features);
    auto imp_coverage = model.get_feature_importance_coverage(n_features);
    auto imp_full     = model.get_feature_importance_full(n_features);
    auto imp_raw      = model.get_feature_importance(n_features);

    failures += test_assert(imp_gain.size() == static_cast<size_t>(n_features),
                            "gain importance vector size == num_features");
    failures += test_assert(imp_coverage.size() == static_cast<size_t>(n_features),
                            "coverage importance vector size == num_features");
    failures += test_assert(imp_full.frequency.size() == static_cast<size_t>(n_features),
                            "full.frequency vector size == num_features");
    failures += test_assert(imp_full.gain.size() == static_cast<size_t>(n_features),
                            "full.gain vector size == num_features");
    failures += test_assert(imp_full.coverage.size() == static_cast<size_t>(n_features),
                            "full.coverage vector size == num_features");

    for (int i = 0; i < n_features; ++i) {
        failures += test_assert(imp_gain[i] >= 0.0f,
                                ("gain[" + std::to_string(i) + "] >= 0").c_str());
        failures += test_assert(imp_coverage[i] >= 0.0f,
                                ("coverage[" + std::to_string(i) + "] >= 0").c_str());
        failures += test_assert(imp_full.frequency[i] >= 0.0f,
                                ("full.frequency[" + std::to_string(i) + "] >= 0").c_str());
        failures += test_assert(imp_full.gain[i] >= 0.0f,
                                ("full.gain[" + std::to_string(i) + "] >= 0").c_str());
        failures += test_assert(imp_full.coverage[i] >= 0.0f,
                                ("full.coverage[" + std::to_string(i) + "] >= 0").c_str());
    }

    float gain_sum = 0.0f;
    float cov_sum  = 0.0f;
    float freq_sum = 0.0f;
    float fgain_sum = 0.0f;
    float fcov_sum  = 0.0f;
    float ffreq_sum = 0.0f;
    for (int i = 0; i < n_features; ++i) {
        gain_sum  += imp_gain[i];
        cov_sum   += imp_coverage[i];
        freq_sum  += imp_full.frequency[i];
        fgain_sum += imp_full.gain[i];
        fcov_sum  += imp_full.coverage[i];
        ffreq_sum += imp_full.frequency[i];
    }

    failures += test_assert_approx(gain_sum, 1.0f,  "gain importance sum ≈ 1.0");
    failures += test_assert_approx(cov_sum,  1.0f,  "coverage importance sum ≈ 1.0");
    failures += test_assert_approx(freq_sum, 1.0f,  "full.frequency sum ≈ 1.0");
    failures += test_assert_approx(fgain_sum, 1.0f, "full.gain sum ≈ 1.0");
    failures += test_assert_approx(fcov_sum,  1.0f, "full.coverage sum ≈ 1.0");

    float raw_freq_total = 0.0f;
    for (float v : imp_raw) raw_freq_total += v;
    if (raw_freq_total > 0.0f) {
        for (int i = 0; i < n_features; ++i) {
            float norm_raw = imp_raw[i] / raw_freq_total;
            failures += test_assert_approx(norm_raw, imp_full.frequency[i],
                                           ("full.frequency[" + std::to_string(i) + "] matches normalized get_feature_importance").c_str());
        }
    }

    if (failures == 0)
        std::cout << "  => ALL PASS (0 failures)\n";
    return failures;
}

int test_inference_engine_consistency() {
    int failures = 0;
    std::cout << "Test 9: Inference Engine consistency\n";

    const int n_samples  = 500;
    const int n_features = 5;
    const int n_train    = 400;

    auto X = torch::randn({n_samples, n_features});
    auto y = X.index({"...", 0})
           + 0.5f * X.index({"...", 1})
           + 0.25f * X.index({"...", 2})
           + 0.1f * torch::randn({n_samples});

    auto X_train = X.index({torch::indexing::Slice(0, n_train)});
    auto y_train = y.index({torch::indexing::Slice(0, n_train)});

    gbdt::GBDTConfig config;
    config.num_trees        = 10;
    config.max_depth        = 4;
    config.min_samples_leaf = 5;
    config.learning_rate    = 0.1f;

    gbdt::GBDT model(config);
    model.fit(X_train, y_train);

    auto pred_normal = model.predict(X_train);
    auto pred_batch  = model.predict_batch(X_train);

    failures += test_assert(pred_normal.sizes() == pred_batch.sizes(),
                            "prediction shapes match");

    int64_t N = pred_normal.size(0);
    float max_diff = 0.0f;
    for (int64_t i = 0; i < N; ++i) {
        float diff = std::fabs(pred_normal[i].item<float>() - pred_batch[i].item<float>());
        if (diff > max_diff) max_diff = diff;
    }

    failures += test_assert(max_diff < 1e-5f,
                            "predict() and predict_batch() are consistent (max diff < 1e-5)");

    if (failures == 0)
        std::cout << "  => ALL PASS (0 failures)\n";
    return failures;
}

int test_inference_engine_single_sample() {
    int failures = 0;
    std::cout << "Test 10: Inference Engine single-sample predict\n";

    std::vector<gbdt::TreeNode> tree(3);
    tree[0].feature_idx  = 0;
    tree[0].split_value  = 0.5f;
    tree[0].left_child   = 1;
    tree[0].right_child  = 2;
    tree[0].num_samples  = 100;

    tree[1].leaf_value   = -0.3f;
    tree[1].num_samples  = 50;

    tree[2].leaf_value   = 0.5f;
    tree[2].num_samples  = 50;

    auto flat_tree = gbdt::InferenceEngine::convert_tree(tree);
    failures += test_assert(flat_tree.size() == 3, "flat tree has 3 nodes");
    failures += test_assert(!flat_tree[0].is_leaf(), "root is internal node");
    failures += test_assert(flat_tree[1].is_leaf(), "node 1 is leaf");
    failures += test_assert(flat_tree[2].is_leaf(), "node 2 is leaf");

    float x_left[] = {0.3f};
    float pred_left = gbdt::InferenceEngine::predict_single(x_left, 1, flat_tree);
    failures += test_assert_approx(pred_left, -0.3f, "value <= 0.5 goes left (leaf -0.3)");

    float x_right[] = {0.7f};
    float pred_right = gbdt::InferenceEngine::predict_single(x_right, 1, flat_tree);
    failures += test_assert_approx(pred_right, 0.5f, "value > 0.5 goes right (leaf 0.5)");

    std::vector<gbdt::TreeNode> empty_tree;
    auto empty_flat = gbdt::InferenceEngine::convert_tree(empty_tree);
    failures += test_assert(empty_flat.empty(), "empty tree converts to empty flat");
    float pred_empty = gbdt::InferenceEngine::predict_single(x_left, 1, empty_flat);
    failures += test_assert_approx(pred_empty, 0.0f, "empty flat tree returns 0.0");

    if (failures == 0)
        std::cout << "  => ALL PASS (0 failures)\n";
    return failures;
}

}  // namespace gbdt_test

int main() {
    std::cout << "\n";
    std::cout << "============================================\n";
    std::cout << "  GBDT Core — C++ Unit Tests\n";
    std::cout << "============================================\n";
    std::cout << "\n";

    int total_failures = 0;

    total_failures += gbdt_test::test_treenode_structure();      std::cout << "\n";
    total_failures += gbdt_test::test_histogram_builder();       std::cout << "\n";
    total_failures += gbdt_test::test_split_finder();            std::cout << "\n";
    total_failures += gbdt_test::test_tree_builder();            std::cout << "\n";
    total_failures += gbdt_test::test_gbdt_training();           std::cout << "\n";
    total_failures += gbdt_test::test_json_serialization();      std::cout << "\n";
    total_failures += gbdt_test::test_feature_importance();      std::cout << "\n";
    total_failures += gbdt_test::test_feature_importance_extended(); std::cout << "\n";
    total_failures += gbdt_test::test_column_subsampling();      std::cout << "\n";
    total_failures += gbdt_test::test_inference_engine_consistency(); std::cout << "\n";
    total_failures += gbdt_test::test_inference_engine_single_sample(); std::cout << "\n";

    std::cout << "============================================\n";
    if (total_failures == 0) {
        std::cout << "  RESULT: ALL TESTS PASSED\n";
    } else {
        std::cout << "  RESULT: " << total_failures
                  << " ASSERTION(S) FAILED\n";
    }
    std::cout << "============================================\n";

    return total_failures > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
