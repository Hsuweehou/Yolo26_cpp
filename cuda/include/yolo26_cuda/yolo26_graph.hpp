#pragma once

#include <optional>
#include <string>
#include <vector>

#include "layer_spec.hpp"
#include "types.hpp"

namespace y26::cuda_graph {

/// 自 `cfg/models/26/*.yaml` 解析并应用缩放的 YOLO26 结构描述（不加载权重、不做前向卷积计算）。
class Yolo26Graph {
   public:
    Yolo26Graph() = default;

    /// 从文件加载；`scale_letter` 在存在 `scales:` 时使用，否则可配合 `force_scale` 覆盖。
    static std::optional<Yolo26Graph> fromYamlFile(const std::string& path,
                                                    ScaleLetter scale_letter = ScaleLetter::kN,
                                                    const std::optional<ScaleCoefficients>& force_scale = std::nullopt);

    [[nodiscard]] const Yolo26YamlHeader& header() const noexcept { return header_; }
    /// backbone + head 展平，一行一个 module（与 YAML 中 layer 下标一致）。
    [[nodiscard]] const std::vector<LayerRow>& flat() const noexcept { return flat_; }

    /// 在给定输入分辨率下做 NCHW 形状推理，返回每一层**主输出**（Detect/Segment 等保留多路语义见 shapes 注释）。
    [[nodiscard]] std::vector<BchwShape> infer_shapes(int64_t input_w, int64_t input_h, int64_t batch = 1) const;

    /// 供调试：人类可读层表（含缩放后参数与推断形状）。
    [[nodiscard]] std::string dumpTable(int64_t input_w = 640, int64_t input_h = 640, int64_t batch = 1) const;

   private:
    Yolo26YamlHeader header_{};
    std::vector<LayerRow> flat_{};
    ScaleCoefficients applied_scale_{0.5, 0.25, 1024};

    void flattenAndApplyScale(ScaleLetter letter, const std::optional<ScaleCoefficients>& force);
};

/// 与 PyTorch/Ultralytics 类似的 `make_divisible` 与通道裁剪。
int make_divisible(int x, int divisor, int max_channels) noexcept;
ScaleCoefficients scale_letter_to_coeff(ScaleLetter letter, const Yolo26YamlHeader& h);

}  // namespace y26::cuda_graph
