#include <gbdt/tree.h>
#include <gbdt/json_utils.h>

#include <sstream>
#include <stdexcept>

namespace gbdt {

std::string tree_to_json(const std::vector<TreeNode>& nodes, int num_features) {
    std::ostringstream os;
    JsonWriter w(os);

    os << "{";
    w.indent(1);
    w.write_string("num_features");
    os << ": ";
    w.write_int(num_features);
    os << ",";
    w.indent(1);
    w.write_string("nodes");
    os << ": [";

    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        w.indent(2);
        os << "{";

        w.indent(3);
        w.write_string("feature_idx");
        os << ": ";
        w.write_int(n.feature_idx);
        os << ",";

        w.indent(3);
        w.write_string("split_value");
        os << ": ";
        w.write_float(n.split_value);
        os << ",";

        w.indent(3);
        w.write_string("leaf_value");
        os << ": ";
        w.write_float(n.leaf_value);
        os << ",";

        w.indent(3);
        w.write_string("left_child");
        os << ": ";
        w.write_int(n.left_child);
        os << ",";

        w.indent(3);
        w.write_string("right_child");
        os << ": ";
        w.write_int(n.right_child);
        os << ",";

        w.indent(3);
        w.write_string("depth");
        os << ": ";
        w.write_int(n.depth);
        os << ",";

        w.indent(3);
        w.write_string("num_samples");
        os << ": ";
        w.write_int(n.num_samples);
        os << ",";

        w.indent(3);
        w.write_string("gain");
        os << ": ";
        w.write_float(n.gain);
        os << ",";

        w.indent(3);
        w.write_string("sum_grad");
        os << ": ";
        w.write_float(n.sum_grad);
        os << ",";

        w.indent(3);
        w.write_string("sum_hess");
        os << ": ";
        w.write_float(n.sum_hess);
        os << ",";

        w.indent(3);
        w.write_string("default_left");
        os << ": ";
        w.write_bool(n.default_left);
        os << ",";

        w.indent(3);
        w.write_string("is_leaf");
        os << ": ";
        w.write_bool(n.is_leaf());

        w.indent(2);
        os << "}";
        if (i + 1 < nodes.size())
            os << ",";
    }

    w.indent(1);
    os << "]";
    w.indent(0);
    os << "}\n";

    return os.str();
}

std::vector<TreeNode> tree_from_json(const std::string& json_str) {
    if (json_str.empty())
        return {};

    JsonParser parser(json_str);

    parser.expect('{');

    while (parser.peek() != '}') {
        std::string key = parser.parse_string();
        parser.expect(':');

        if (key == "num_features") {
            parser.parse_int();
        } else if (key == "nodes") {
            std::vector<TreeNode> nodes = parser.parse_tree_nodes();

            while (parser.peek() != '}') {
                if (parser.peek() == ',')
                    parser.consume();
                std::string remaining_key = parser.parse_string();
                parser.expect(':');
                parser.skip_value();
            }
            parser.expect('}');
            return nodes;
        } else {
            parser.skip_value();
        }

        if (parser.peek() == ',')
            parser.consume();
    }

    parser.expect('}');
    return {};
}

}
