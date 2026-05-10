#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace y26::cuda_graph {

/// NCHW 形状（推理排布常用）；batch 可为导出时的符号维度 1。
struct BchwShape {
    int64_t n = 1;
    int64_t c = 0;
    int64_t h = 0;
    int64_t w = 0;

    [[nodiscard]] bool operator==(const BchwShape& o) const noexcept {
        return n == o.n && c == o.c && h == o.h && w == o.w;
    }
};

/// Ultralytics 风格复合缩放：[depth, width, max_channels]
struct ScaleCoefficients {
    double depth = 0.5;
    double width = 0.25;
    int max_channels = 1024;
};

enum class ScaleLetter { kN, kS, kM, kL, kX };

}  // namespace y26::cuda_graph
