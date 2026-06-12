#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/torch.h>
#include <gbdt/gbdt.h>
#include <string>

namespace py = pybind11;

// Helper: convert numpy array to torch tensor
torch::Tensor numpy_to_tensor(py::array_t<float> array) {
    py::buffer_info buf = array.request();
    std::vector<int64_t> shape(buf.shape.begin(), buf.shape.end());
    auto device = torch::kCPU;
    return torch::from_blob(buf.ptr, shape, torch::kFloat32).clone().to(device);
}

py::array_t<float> tensor_to_numpy(const torch::Tensor& tensor) {
    auto owned = std::make_unique<torch::Tensor>(tensor.contiguous().to(torch::kCPU));
    float* data_ptr = owned->data_ptr<float>();
    std::vector<py::ssize_t> shape(owned->sizes().begin(), owned->sizes().end());
    auto capsule = py::capsule(owned.release(), [](void* p) {
        delete static_cast<torch::Tensor*>(p);
    });
    return py::array_t<float>(shape, data_ptr, capsule);
}

PYBIND11_MODULE(gbdt_python, m) {
    m.doc() = "GBDT for Quantitative Investment - C++ core with pybind11";

    // GBDTConfig
    py::class_<gbdt::GBDTConfig>(m, "GBDTConfig")
        .def(py::init<>())
        .def_readwrite("max_depth", &gbdt::GBDTConfig::max_depth)
        .def_readwrite("min_samples_leaf", &gbdt::GBDTConfig::min_samples_leaf)
        .def_readwrite("min_gain_to_split", &gbdt::GBDTConfig::min_gain_to_split)
        .def_readwrite("max_bins", &gbdt::GBDTConfig::max_bins)
        .def_readwrite("lambda_l2", &gbdt::GBDTConfig::lambda_l2)
        .def_readwrite("lambda_l1", &gbdt::GBDTConfig::lambda_l1)
        .def_readwrite("gamma", &gbdt::GBDTConfig::gamma)
        .def_readwrite("num_trees", &gbdt::GBDTConfig::num_trees)
        .def_readwrite("learning_rate", &gbdt::GBDTConfig::learning_rate)
        .def_readwrite("subsample_row", &gbdt::GBDTConfig::subsample_row)
        .def_readwrite("subsample_col", &gbdt::GBDTConfig::subsample_col)
        .def_readwrite("loss_type", &gbdt::GBDTConfig::loss_type)
        .def_readwrite("early_stopping_rounds", &gbdt::GBDTConfig::early_stopping_rounds)
        .def_readwrite("early_stopping_tol", &gbdt::GBDTConfig::early_stopping_tol)
        .def_readwrite("use_line_search", &gbdt::GBDTConfig::use_line_search)
        .def_readwrite("enable_missing_values", &gbdt::GBDTConfig::enable_missing_values)
        .def_readwrite("huber_delta", &gbdt::GBDTConfig::huber_delta)
        .def_readwrite("mae_eps", &gbdt::GBDTConfig::mae_eps)
        .def_readwrite("num_threads", &gbdt::GBDTConfig::num_threads)
        .def_readwrite("random_seed", &gbdt::GBDTConfig::random_seed)
        .def("__repr__", [](const gbdt::GBDTConfig& c) {
            return "<GBDTConfig num_trees=" + std::to_string(c.num_trees)
                + " max_depth=" + std::to_string(c.max_depth)
                + " lr=" + std::to_string(c.learning_rate) + ">";
        });

    // MonotoneConstraint
    py::class_<gbdt::MonotoneConstraint>(m, "MonotoneConstraint")
        .def(py::init<>())
        .def_readwrite("feature_idx", &gbdt::MonotoneConstraint::feature_idx)
        .def_readwrite("dir", &gbdt::MonotoneConstraint::dir);

    py::enum_<gbdt::MonotoneConstraint::Direction>(m, "MonotoneDirection")
        .value("INCREASING", gbdt::MonotoneConstraint::INCREASING)
        .value("DECREASING", gbdt::MonotoneConstraint::DECREASING)
        .value("NONE", gbdt::MonotoneConstraint::NONE);

    // TreeNode — exposed for manual training loops
    py::class_<gbdt::TreeNode>(m, "TreeNode")
        .def(py::init<>())
        .def_readwrite("feature_idx", &gbdt::TreeNode::feature_idx)
        .def_readwrite("split_value", &gbdt::TreeNode::split_value)
        .def_readwrite("left_child", &gbdt::TreeNode::left_child)
        .def_readwrite("right_child", &gbdt::TreeNode::right_child)
        .def_readwrite("leaf_value", &gbdt::TreeNode::leaf_value)
        .def_readwrite("num_samples", &gbdt::TreeNode::num_samples)
        .def_readwrite("gain", &gbdt::TreeNode::gain)
        .def_readwrite("sum_grad", &gbdt::TreeNode::sum_grad)
        .def_readwrite("sum_hess", &gbdt::TreeNode::sum_hess)
        .def_readwrite("depth", &gbdt::TreeNode::depth)
        .def_readwrite("default_left", &gbdt::TreeNode::default_left)
        .def("is_leaf", &gbdt::TreeNode::is_leaf)
        .def("__repr__", [](const gbdt::TreeNode& n) {
            if (n.is_leaf())
                return "<TreeNode leaf value=" + std::to_string(n.leaf_value) + ">";
            return "<TreeNode feat=" + std::to_string(n.feature_idx)
                + " thr=" + std::to_string(n.split_value) + ">";
        });

    // GBDT main class
    py::class_<gbdt::GBDT>(m, "GBDT")
        .def(py::init<const gbdt::GBDTConfig&>(), py::arg("config"))
        .def("fit", [](gbdt::GBDT& model,
                       py::array_t<float> X,
                       py::array_t<float> y,
                       py::array_t<float> X_val,
                       py::array_t<float> y_val,
                       const std::vector<gbdt::MonotoneConstraint>& constraints) {
            auto X_tensor = numpy_to_tensor(X);
            auto y_tensor = numpy_to_tensor(y);
            auto X_val_tensor = numpy_to_tensor(X_val);
            auto y_val_tensor = numpy_to_tensor(y_val);
            model.fit(X_tensor, y_tensor, X_val_tensor, y_val_tensor, constraints);
        }, py::arg("X"), py::arg("y"),
           py::arg("X_val") = py::array_t<float>(),
           py::arg("y_val") = py::array_t<float>(),
           py::arg("constraints") = std::vector<gbdt::MonotoneConstraint>())
        .def("fit_one_tree", [](gbdt::GBDT& model,
                                 py::array_t<float> X,
                                 py::array_t<float> gradients,
                                 py::array_t<float> hessians,
                                 const std::vector<gbdt::MonotoneConstraint>& constraints) {
            auto X_tensor = numpy_to_tensor(X);
            auto grad_tensor = numpy_to_tensor(gradients);
            auto hess_tensor = numpy_to_tensor(hessians);
            return model.fit_one_tree(X_tensor, grad_tensor, hess_tensor, constraints);
        }, py::arg("X"), py::arg("gradients"), py::arg("hessians"),
           py::arg("constraints") = std::vector<gbdt::MonotoneConstraint>())
        .def("predict", [](gbdt::GBDT& model, py::array_t<float> X) {
            auto X_tensor = numpy_to_tensor(X);
            auto result = model.predict(X_tensor);
            return tensor_to_numpy(result);
        }, py::arg("X"))
        .def("predict_tree", [](gbdt::GBDT& model,
                                 const std::vector<gbdt::TreeNode>& tree,
                                 py::array_t<float> X) {
            auto X_tensor = numpy_to_tensor(X);
            auto result = model.predict_tree(tree, X_tensor);
            return tensor_to_numpy(result);
        }, py::arg("tree"), py::arg("X"))
        .def("num_trees", &gbdt::GBDT::num_trees)
        .def("to_json", &gbdt::GBDT::to_json)
        .def("from_json", &gbdt::GBDT::from_json)
        .def("set_state", &gbdt::GBDT::set_state,
             py::arg("init_pred"), py::arg("trees"), py::arg("tree_step_sizes"))
        .def("get_feature_importance", &gbdt::GBDT::get_feature_importance,
             py::arg("num_features"))
        .def("get_feature_importance_gain", &gbdt::GBDT::get_feature_importance_gain,
             py::arg("num_features"))
        .def("get_feature_importance_coverage", &gbdt::GBDT::get_feature_importance_coverage,
             py::arg("num_features"))
        .def("get_feature_importance_full", [](gbdt::GBDT& model, int num_features) {
            auto imp = model.get_feature_importance_full(num_features);
            py::dict d;
            d["frequency"] = imp.frequency;
            d["gain"] = imp.gain;
            d["coverage"] = imp.coverage;
            return d;
        }, py::arg("num_features"))
        .def_property_readonly("train_losses", [](gbdt::GBDT& model) {
            return model.get_metrics().train_losses;
        })
        .def_property_readonly("val_losses", [](gbdt::GBDT& model) {
            return model.get_metrics().val_losses;
        })
        .def("__repr__", [](const gbdt::GBDT& model) {
            return "<GBDT model with " + std::to_string(model.num_trees()) + " trees>";
        });
}
