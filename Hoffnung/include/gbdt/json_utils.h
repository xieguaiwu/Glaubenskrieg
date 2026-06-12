#pragma once
#include "tree.h"
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gbdt {

class JsonWriter {
public:
    explicit JsonWriter(std::ostringstream& os) : os_(os) {}

    void write_null() { os_ << "null"; }

    void write_bool(bool v) { os_ << (v ? "true" : "false"); }

    void write_int(int v) { os_ << v; }

    void write_float(float v) {
        if (std::isnan(v)) {
            os_ << "null";
        } else if (std::isinf(v)) {
            os_ << (v > 0 ? "1e308" : "-1e308");
        } else {
            os_ << std::setprecision(9) << v;
        }
    }

    void write_string(const std::string& s) {
        os_ << '"';
        for (char c : s) {
            switch (c) {
                case '"':  os_ << "\\\""; break;
                case '\\': os_ << "\\\\"; break;
                case '\n': os_ << "\\n";  break;
                case '\r': os_ << "\\r";  break;
                case '\t': os_ << "\\t";  break;
                default:   os_ << c;
            }
        }
        os_ << '"';
    }

    void indent(int level) {
        os_ << '\n' << std::string(static_cast<size_t>(level) * 2, ' ');
    }

    std::ostringstream& stream() { return os_; }

private:
    std::ostringstream& os_;
};

class JsonParser {
public:
    explicit JsonParser(const std::string& input) : input_(input), pos_(0) {}

    void skip_ws() {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_])))
            ++pos_;
    }

    char peek() {
        skip_ws();
        return pos_ < input_.size() ? input_[pos_] : '\0';
    }

    char consume() {
        skip_ws();
        if (pos_ >= input_.size())
            throw std::runtime_error("Unexpected end of JSON");
        return input_[pos_++];
    }

    void expect(char c) {
        char actual = consume();
        if (actual != c) {
            std::string msg = "Expected '";
            msg += c;
            msg += "' but got '";
            msg += actual;
            msg += "'";
            throw std::runtime_error(msg);
        }
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (pos_ < input_.size()) {
            char c = input_[pos_++];
            if (c == '"') return result;
            if (c == '\\') {
                if (pos_ >= input_.size()) break;
                char esc = input_[pos_++];
                switch (esc) {
                    case '"':  result += '"';  break;
                    case '\\': result += '\\'; break;
                    case '/':  result += '/';  break;
                    case 'n':  result += '\n'; break;
                    case 'r':  result += '\r'; break;
                    case 't':  result += '\t'; break;
                    case 'u': {
                        if (pos_ + 4 <= input_.size()) {
                            char hex[5] = {};
                            for (int i = 0; i < 4; ++i) {
                                hex[i] = input_[pos_++];
                            }
                            char* end = nullptr;
                            unsigned long codepoint = std::strtoul(hex, &end, 16);
                            if (end != hex + 4)
                                throw std::runtime_error("Invalid unicode escape");

                            // Encode as UTF-8 (BMP only, surrogates -> replacement char)
                            if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
                                // Surrogate pair half — emit U+FFFD replacement character
                                result += "\xEF\xBF\xBD";
                            } else if (codepoint <= 0x7F) {
                                result += static_cast<char>(codepoint);
                            } else if (codepoint <= 0x7FF) {
                                result += static_cast<char>(0xC0 | (codepoint >> 6));
                                result += static_cast<char>(0x80 | (codepoint & 0x3F));
                            } else {
                                result += static_cast<char>(0xE0 | (codepoint >> 12));
                                result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                                result += static_cast<char>(0x80 | (codepoint & 0x3F));
                            }
                        }
                        break;
                    }
                    default: result += esc;
                }
            } else {
                result += c;
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    // parse_number does NOT silently catch exceptions — they propagate.
    float parse_number() {
        skip_ws();
        // Handle JSON null as NaN
        if (pos_ + 4 <= input_.size() && input_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return std::numeric_limits<float>::quiet_NaN();
        }
        size_t start = pos_;
        if (pos_ < input_.size() && input_[pos_] == '-') ++pos_;
        while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_])))
            ++pos_;
        if (pos_ < input_.size() && input_[pos_] == '.') {
            ++pos_;
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_])))
                ++pos_;
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-'))
                ++pos_;
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_])))
                ++pos_;
        }
        return std::stof(input_.substr(start, pos_ - start));
    }

    int parse_int() {
        float v = parse_number();
        if (std::isnan(v))
            throw std::runtime_error("Integer field cannot be null/NaN");
        return static_cast<int>(v);
    }

    bool parse_bool() {
        skip_ws();
        if (pos_ + 4 <= input_.size() && input_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            return true;
        }
        if (pos_ + 5 <= input_.size() && input_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            return false;
        }
        throw std::runtime_error("Expected boolean (true/false)");
    }

    std::vector<TreeNode> parse_tree_nodes() {
        expect('[');
        std::vector<TreeNode> nodes;
        if (peek() == ']') {
            consume();
            return nodes;
        }
        while (true) {
            nodes.push_back(parse_tree_node());
            if (peek() == ',') {
                consume();
                continue;
            }
            break;
        }
        expect(']');

        int max_idx = static_cast<int>(nodes.size()) - 1;
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];
            int l = node.left_child;
            int r = node.right_child;
            if ((l >= 0 && l > max_idx) || (r >= 0 && r > max_idx)) {
                throw std::runtime_error(
                    "Tree node " + std::to_string(i) +
                    " child index out of bounds: left=" + std::to_string(l) +
                    " right=" + std::to_string(r) +
                    " max=" + std::to_string(max_idx));
            }
        }

        return nodes;
    }

    TreeNode parse_tree_node() {
        expect('{');
        TreeNode node;
        while (true) {
            if (peek() == '}') {
                consume();
                break;
            }
            std::string key = parse_string();
            expect(':');
            if (key == "feature_idx") node.feature_idx = parse_int();
            else if (key == "split_value") node.split_value = parse_number();
            else if (key == "leaf_value") node.leaf_value = parse_number();
            else if (key == "left_child") node.left_child = parse_int();
            else if (key == "right_child") node.right_child = parse_int();
            else if (key == "depth") node.depth = parse_int();
            else if (key == "num_samples") node.num_samples = parse_int();
            else if (key == "gain") node.gain = parse_number();
            else if (key == "sum_grad") node.sum_grad = parse_number();
            else if (key == "sum_hess") node.sum_hess = parse_number();
            else if (key == "default_left") node.default_left = parse_bool();
            else if (key == "is_leaf") parse_bool();
            else skip_value();
            if (peek() == ',') consume();
        }
        return node;
    }

    void skip_value() {
        skip_ws();
        if (pos_ >= input_.size()) return;
        char c = input_[pos_];
        if (c == '"') { parse_string(); }
        else if (c == 't' || c == 'f') { parse_bool(); }
        else if (c == 'n') { expect('n'); expect('u'); expect('l'); expect('l'); }
        else if (c == '[') {
            ++pos_;
            int depth = 1;
            while (depth > 0 && pos_ < input_.size()) {
                char cc = input_[pos_];
                if (cc == '"') { parse_string(); continue; }
                else if (cc == '[') ++depth;
                else if (cc == ']') --depth;
                ++pos_;
            }
        } else if (c == '{') {
            ++pos_;
            int depth = 1;
            while (depth > 0 && pos_ < input_.size()) {
                char cc = input_[pos_];
                if (cc == '"') { parse_string(); continue; }
                else if (cc == '{') ++depth;
                else if (cc == '}') --depth;
                ++pos_;
            }
        } else { parse_number(); }
    }

private:
    const std::string& input_;
    size_t pos_;
};

} // namespace gbdt
