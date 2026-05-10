#include "yolo26_cuda/yolo26_graph.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <optional>
#include <sstream>

#include <yaml-cpp/yaml.h>

namespace y26::cuda_graph {

namespace {

int64_t arg_as_int64(const LayerArg& a) {
    if (const auto* i = std::get_if<int64_t>(&a)) {
        return *i;
    }
    if (const auto* d = std::get_if<double>(&a)) {
        return static_cast<int64_t>(std::lround(*d));
    }
    return 0;
}

double arg_as_double(const LayerArg& a) {
    if (const auto* d = std::get_if<double>(&a)) {
        return *d;
    }
    if (const auto* i = std::get_if<int64_t>(&a)) {
        return static_cast<double>(*i);
    }
    return 0.0;
}

bool arg_is_number(const LayerArg& a) {
    return std::holds_alternative<int64_t>(a) || std::holds_alternative<double>(a);
}

LayerArg parse_arg_node(const YAML::Node& n) {
    if (!n.IsDefined() || n.IsNull()) {
        return std::monostate{};
    }
    if (n.IsScalar()) {
        if (n.Tag() == "!!bool" || n.Type() == YAML::NodeType::Scalar) {
            try {
                if (n.as<std::string>() == "True" || n.as<std::string>() == "true" ||
                    n.as<std::string>() == "False" || n.as<std::string>() == "false") {
                    return LayerArg{n.as<bool>()};
                }
            } catch (...) {
            }
        }
        const std::string s = n.as<std::string>();
        if (s == "nc" || s == "kpt_shape") {
            return LayerArg{s};
        }
        if (s == "True" || s == "true") {
            return true;
        }
        if (s == "False" || s == "false") {
            return false;
        }
        try {
            if (s.find('.') != std::string::npos) {
                return LayerArg{std::stod(s)};
            }
            return LayerArg{static_cast<int64_t>(std::stoll(s))};
        } catch (...) {
            return LayerArg{s};
        }
    }
    if (n.IsSequence()) {
        std::ostringstream os;
        os << "[";
        for (std::size_t i = 0; i < n.size(); ++i) {
            if (i) {
                os << ",";
            }
            if (n[i].IsScalar()) {
                os << n[i].as<std::string>();
            } else {
                os << "?";
            }
        }
        os << "]";
        return LayerArg{os.str()};
    }
    return std::monostate{};
}

std::vector<int> parse_from_node(const YAML::Node& f) {
    std::vector<int> out;
    if (f.IsSequence()) {
        for (const auto& e : f) {
            out.push_back(e.as<int>());
        }
    } else {
        out.push_back(f.as<int>());
    }
    return out;
}

LayerRow parse_one_row(const YAML::Node& row) {
    LayerRow lr;
    if (!row.IsSequence() || row.size() < 4) {
        return lr;
    }
    lr.from = parse_from_node(row[0]);
    lr.repeats = row[1].as<int>();
    lr.module = row[2].as<std::string>();
    const YAML::Node args = row[3];
    if (args.IsSequence()) {
        for (const auto& a : args) {
            lr.args.push_back(parse_arg_node(a));
        }
    }
    return lr;
}

void parse_layer_block(const YAML::Node& n, std::vector<LayerRow>& out) {
    if (!n || !n.IsSequence()) {
        return;
    }
    for (const auto& row : n) {
        out.push_back(parse_one_row(row));
    }
}

bool read_top_level(const YAML::Node& root, Yolo26YamlHeader& h) {
    if (root["nc"]) {
        h.nc = root["nc"].as<int>();
    }
    if (root["reg_max"]) {
        h.reg_max = root["reg_max"].as<int>();
    }
    if (root["end2end"]) {
        try {
            h.end2end = root["end2end"].as<bool>();
        } catch (...) {
            const std::string e = root["end2end"].as<std::string>();
            h.end2end = (e == "True" || e == "true" || e == "1");
        }
    } else {
        h.end2end = true;
    }
    if (root["text_model"]) {
        h.text_model = root["text_model"].as<std::string>();
    }
    h.kpt_shape = {17, 3};
    if (root["kpt_shape"] && root["kpt_shape"].IsSequence()) {
        h.kpt_shape.clear();
        for (const auto& e : root["kpt_shape"]) {
            h.kpt_shape.push_back(e.as<int64_t>());
        }
    }
    if (root["scales"]) {
        h.has_scales_block = true;
        for (const auto& kv : root["scales"]) {
            const std::string k = kv.first.as<std::string>();
            if (k.size() != 1U) {
                continue;
            }
            if (!kv.second.IsSequence() || kv.second.size() < 3) {
                continue;
            }
            ScaleCoefficients s{};
            s.depth = kv.second[0].as<double>();
            s.width = kv.second[1].as<double>();
            s.max_channels = kv.second[2].as<int>();
            h.scales_by_name[k] = s;
        }
    }
    parse_layer_block(root["backbone"], h.backbone);
    parse_layer_block(root["head"], h.head);
    return !h.backbone.empty() && !h.head.empty();
}

BchwShape get_tensor_from_ref(int ref, int layer_i, const BchwShape& in0, const std::vector<BchwShape>& built) {
    if (ref == -1) {
        if (layer_i == 0) {
            return in0;
        }
        return built[static_cast<size_t>(layer_i - 1)];
    }
    return built[static_cast<size_t>(ref)];
}

BchwShape conv_out_spatial(const BchwShape& in, int k, int s) {
    BchwShape c = in;
    const int pad = (k - 1) / 2;
    c.h = (in.h + 2 * pad - k) / s + 1;
    c.w = (in.w + 2 * pad - k) / s + 1;
    return c;
}

int detect_output_channels(int nc, int reg_max) { return 4 * reg_max + nc; }

void scale_ch_at(std::vector<LayerArg>& a, size_t idx, const ScaleCoefficients& sc) {
    if (idx >= a.size() || !arg_is_number(a[idx])) {
        return;
    }
    int base = static_cast<int>(arg_as_double(a[idx]));
    int y = static_cast<int>(std::lround(static_cast<double>(base) * sc.width));
    y = (std::min)(y, sc.max_channels);
    a[idx] = static_cast<int64_t>(make_divisible((std::max)(1, y), 8, sc.max_channels));
}

void scale_one_row(LayerRow& lr, const ScaleCoefficients& sc, const Yolo26YamlHeader& h) {
    lr.scaled_repeats = static_cast<int>(std::max(1, (int)std::lround(lr.repeats * sc.depth)));
    lr.scaled_args = lr.args;
    std::vector<LayerArg>& a = lr.scaled_args;
    const std::string& m = lr.module;

    if (m == "nn.Upsample" || m == "Concat") {
        return;
    }
    if (m == "Conv" || m == "C3k2" || m == "SPPF" || m == "C2PSA") {
        if (!a.empty()) {
            scale_ch_at(a, 0, sc);
        }
        return;
    }
    if (m == "Classify") {
        if (!a.empty() && (arg_is_number(a[0]) || (std::holds_alternative<std::string>(a[0]) && std::get<std::string>(a[0]) == "nc"))) {
            a.clear();
        }
        a = {static_cast<int64_t>(h.nc)};
        return;
    }
    if (m == "Segment26") {
        if (a.size() < 2U) {
            return;
        }
        a[0] = static_cast<int64_t>(h.nc);
        for (size_t j = 1; j < a.size(); ++j) {
            scale_ch_at(a, j, sc);
        }
        return;
    }
    if (m == "YOLOEDetect" && a.size() >= 2U) {
        a[0] = static_cast<int64_t>(h.nc);
        scale_ch_at(a, 1, sc);
        return;
    }
    if (m == "Detect" && !a.empty() && !arg_is_number(a[0])) {
        a[0] = static_cast<int64_t>(h.nc);
    } else if (m == "Detect" && a.empty()) {
        a = {static_cast<int64_t>(h.nc)};
    }
    if (m == "OBB26" && !a.empty()) {
        a[0] = static_cast<int64_t>(h.nc);
    }
}

void resolve_meta_strings(LayerRow& lr, Yolo26YamlHeader& h) {
    (void)h;
    for (size_t j = 0; j < lr.scaled_args.size(); ++j) {
        if (const auto* s = std::get_if<std::string>(&lr.scaled_args[j])) {
            if (*s == "nc") {
                lr.scaled_args[j] = static_cast<int64_t>(h.nc);
            }
        }
    }
    if (lr.module == "Pose26" && h.kpt_shape.size() >= 2) {
        lr.scaled_args.clear();
        lr.scaled_args.push_back(static_cast<int64_t>(h.nc));
        lr.scaled_args.push_back(h.kpt_shape[0]);
        lr.scaled_args.push_back(h.kpt_shape[1]);
    }
}
}  // namespace

int make_divisible(int x, int divisor, int max_channels) noexcept {
    (void)max_channels;
    int y = (x + divisor - 1) / divisor * divisor;
    y = (std::min)(y, max_channels);
    return (std::max)(divisor, y);
}

ScaleCoefficients Yolo26YamlHeader::scale_for(ScaleLetter letter) const {
    const char* keys[] = {"n", "s", "m", "l", "x"};
    const int idx = (std::min)(static_cast<int>(letter), 4);
    if (scales_by_name.count(keys[idx])) {
        return scales_by_name.at(keys[idx]);
    }
    return {0.50, 0.25, 1024};
}

ScaleCoefficients scale_letter_to_coeff(ScaleLetter letter, const Yolo26YamlHeader& h) {
    return h.scale_for(letter);
}

void Yolo26Graph::flattenAndApplyScale(ScaleLetter letter, const std::optional<ScaleCoefficients>& force) {
    applied_scale_ = force.value_or(header_.scale_for(letter));
    flat_.clear();
    for (const auto& b : header_.backbone) {
        flat_.push_back(b);
    }
    for (const auto& t : header_.head) {
        flat_.push_back(t);
    }
    for (auto& r : flat_) {
        scale_one_row(r, applied_scale_, header_);
        resolve_meta_strings(r, header_);
    }
}

std::optional<Yolo26Graph> Yolo26Graph::fromYamlFile(const std::string& path, ScaleLetter scale_letter,
    const std::optional<ScaleCoefficients>& force_scale) {
    try {
        const YAML::Node root = YAML::LoadFile(path);
        Yolo26Graph g;
        g.header_.source_path = path;
        if (!read_top_level(root, g.header_)) {
            return std::nullopt;
        }
        g.flattenAndApplyScale(scale_letter, force_scale);
        return g;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

static int resolve_ref(int from_val, int layer_i) {
    if (from_val == -1) {
        if (layer_i == 0) {
            return -1;
        }
        return layer_i - 1;
    }
    return from_val;
}

std::vector<BchwShape> Yolo26Graph::infer_shapes(int64_t input_w, int64_t input_h, int64_t batch) const {
    std::vector<BchwShape> outs;
    if (flat_.empty()) {
        return outs;
    }
    const int n_layers = static_cast<int>(flat_.size());
    outs.assign(static_cast<size_t>(n_layers), BchwShape{});
    BchwShape in0;
    in0.n = batch;
    in0.c = 3;
    in0.h = input_h;
    in0.w = input_w;

    for (int i = 0; i < n_layers; ++i) {
        const LayerRow& r = flat_[static_cast<size_t>(i)];
        const std::string& m = r.module;
        if (m == "Conv") {
            const int ref0 = resolve_ref(r.from[0], i);
            BchwShape x = get_tensor_from_ref(ref0, i, in0, outs);
            const int k = r.scaled_args.size() > 1U ? (int)arg_as_int64(r.scaled_args[1]) : 3;
            const int s = r.scaled_args.size() > 2U ? (int)arg_as_int64(r.scaled_args[2]) : 1;
            BchwShape o = conv_out_spatial(x, k, s);
            o.c = r.scaled_args.empty() ? x.c : arg_as_int64(r.scaled_args[0]);
            outs[static_cast<size_t>(i)] = o;
        } else if (m == "C3k2" || m == "SPPF" || m == "C2PSA") {
            const int ref0 = resolve_ref(r.from[0], i);
            BchwShape x = get_tensor_from_ref(ref0, i, in0, outs);
            int64_t c2 = 256;
            if (!r.scaled_args.empty() && arg_is_number(r.scaled_args[0])) {
                c2 = arg_as_int64(r.scaled_args[0]);
            }
            BchwShape o = x;
            o.c = c2;
            outs[static_cast<size_t>(i)] = o;
        } else if (m == "nn.Upsample") {
            const int ref0 = resolve_ref(r.from[0], i);
            BchwShape x = get_tensor_from_ref(ref0, i, in0, outs);
            int f = 2;
            for (const auto& av : r.scaled_args) {
                if (arg_is_number(av) && arg_as_int64(av) > 1) {
                    f = (int)arg_as_int64(av);
                }
            }
            BchwShape o = x;
            o.h = x.h * f;
            o.w = x.w * f;
            outs[static_cast<size_t>(i)] = o;
        } else if (m == "Concat") {
            const int r0 = resolve_ref(r.from[0], i);
            BchwShape x = get_tensor_from_ref(r0, i, in0, outs);
            int64_t csum = x.c;
            const int64_t hh = x.h;
            const int64_t ww = x.w;
            for (size_t k = 1; k < r.from.size(); ++k) {
                const int rj = resolve_ref(r.from[k], i);
                BchwShape t = get_tensor_from_ref(rj, i, in0, outs);
                csum += t.c;
                (void)hh;
                (void)ww;
            }
            BchwShape o = x;
            o.c = csum;
            o.h = hh;
            o.w = ww;
            outs[static_cast<size_t>(i)] = o;
        } else if (m == "Classify" || m == "Detect" || m == "Segment26" || m == "Pose26" || m == "OBB26" ||
                   m == "YOLOEDetect") {
            const int ref0 = resolve_ref(r.from[0], i);
            const BchwShape x0 = get_tensor_from_ref(ref0, i, in0, outs);
            BchwShape t{};
            t.n = batch;
            if (m == "Classify") {
                t.c = header_.nc;
                t.h = 1;
                t.w = 1;
            } else {
                t.h = x0.h;
                t.w = x0.w;
                t.c = static_cast<int64_t>(detect_output_channels(header_.nc, header_.reg_max));
            }
            outs[static_cast<size_t>(i)] = t;
        } else {
            if (!r.from.empty()) {
                const int ref0 = resolve_ref(r.from[0], i);
                if (ref0 == -1) {
                    outs[static_cast<size_t>(i)] = in0;
                } else {
                    outs[static_cast<size_t>(i)] = outs[static_cast<size_t>(ref0)];
                }
            }
        }
    }
    return outs;
}

std::string Yolo26Graph::dumpTable(int64_t input_w, int64_t input_h, int64_t batch) const {
    const std::vector<BchwShape> s = infer_shapes(input_w, input_h, batch);
    std::ostringstream o;
    o << "file: " << header_.source_path << "  nc=" << header_.nc << "  reg_max=" << header_.reg_max << "  e2e="
      << (header_.end2end ? "1" : "0") << "\n\n";
    o << "idx  module            r  from    output (N,C,H,W)\n";
    o << "----------------------------------------------------------------\n";
    for (int i = 0; i < (int)flat_.size() && i < (int)s.size(); ++i) {
        const LayerRow& r = flat_[(size_t)i];
        o << std::setw(3) << i << "  " << std::setw(16) << std::left << r.module;
        o << "  " << std::setw(2) << r.scaled_repeats << "  [";
        for (size_t fi = 0; fi < r.from.size(); ++fi) {
            if (fi) {
                o << ',';
            }
            o << r.from[fi];
        }
        o << "]  (" << s[(size_t)i].n << "," << s[(size_t)i].c << "," << s[(size_t)i].h << "," << s[(size_t)i].w << ")\n";
    }
    return o.str();
}

}  // namespace y26::cuda_graph
