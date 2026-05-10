#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace y26::cuda {

/// 设备指针 `d_p` 上原地 SiLU，长度为 `n` 的 float 元素。`s` 可为 `nullptr`（流 0）。
[[nodiscard]] cudaError_t silu_inplace_f32_device(float* d_p, std::size_t n, cudaStream_t s = nullptr);

}  // namespace y26::cuda
