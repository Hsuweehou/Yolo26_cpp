#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "types.hpp"

namespace y26::cuda_graph {

/// 与 Ultralytics 模型 cfg 中 `args` 序列一一对应（支持 None/布尔/整数/浮点/字符串）。
using LayerArg = std::variant<std::monostate, int64_t, double, bool, std::string>;

/// 单层 YAML 行解析结果：[from, repeats, module, args...]
struct LayerRow {
    /// `from` 原始值：单路为长度 1；Concat 等为多路。其中使用 -1 表示「上一层」，在构图时替换为具体下标。
    std::vector<int> from{};
    int repeats = 1;
    std::string module{};
    std::vector<LayerArg> args{};

    /// 对通道、block 内重复等应用 width/depth/max_channels 后用于形状推理的副本；Detect/Segment 等只补 nc 等，不改变特征图 H/W 逻辑。
    int scaled_repeats = 1;
    std::vector<LayerArg> scaled_args{};
};

struct Yolo26YamlHeader {
    int nc = 80;
    int reg_max = 1;
    bool end2end = true;
    std::string text_model;  // yoloe-26 等可选
    /// 姿态等任务顶层 `kpt_shape: [K, D]`，供 Pose26 行解析参考。
    std::vector<int64_t> kpt_shape{17, 3};
    std::vector<LayerRow> backbone{};
    std::vector<LayerRow> head{};
    std::string source_path{};

    /// `scales:` 块中 n/s/m/l/x 对应 [depth, width, max_channels]；缺省时 `scale_for` 退化为 n 档典型值。
    std::map<std::string, ScaleCoefficients> scales_by_name{};
    bool has_scales_block = false;
    [[nodiscard]] ScaleCoefficients scale_for(ScaleLetter letter) const;
};

}  // namespace y26::cuda_graph
