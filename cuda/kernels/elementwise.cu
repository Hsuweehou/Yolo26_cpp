#include "yolo26_cuda/kernels/elementwise.hpp"

#include <cstddef>
#include <cuda_runtime.h>

namespace y26::cuda {

__global__ static void silu_kernel(float* __restrict__ x, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float t = x[i];
    x[i] = t / (1.0f + __expf(-t));
}

cudaError_t silu_inplace_f32_device(float* d_p, std::size_t n, cudaStream_t s) {
    if (!d_p || n == 0) {
        return cudaSuccess;
    }
    const int tpb = 256;
    const int bl = static_cast<int>((n + static_cast<std::size_t>(tpb) - 1) / static_cast<std::size_t>(tpb));
    silu_kernel<<<bl, tpb, 0, s>>>(d_p, n);
    return cudaGetLastError();
}

}  // namespace y26::cuda
