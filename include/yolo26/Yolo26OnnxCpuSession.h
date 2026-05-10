#ifndef YOLO26_YOLO26ONNXCPUSESSION_H
#define YOLO26_YOLO26ONNXCPUSESSION_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "yolo26/Yolo26VariantBackend.h"

namespace yolo26 {

/// ONNX Runtime CPU 推理后端（与 Yolo26TrtSession 共享 Yolo26VariantBackend 接口）。
class Yolo26OnnxCpuSession : public Yolo26VariantBackend {
public:
    Yolo26OnnxCpuSession();
    ~Yolo26OnnxCpuSession() override;

    Yolo26OnnxCpuSession(const Yolo26OnnxCpuSession&) = delete;
    Yolo26OnnxCpuSession& operator=(const Yolo26OnnxCpuSession&) = delete;

    Yolo26BackendKind kind() const noexcept override { return Yolo26BackendKind::kOnnxRuntime; }

    bool loadFromOnnx(const std::string& onnxPath, std::string_view taskTag) override;

    [[nodiscard]] bool ready() const override;

    [[nodiscard]] int32_t deviceId() const override;
    [[nodiscard]] int modelWidth() const override;
    [[nodiscard]] int modelHeight() const override;

    [[nodiscard]] size_t numInputs() const override;
    [[nodiscard]] size_t numOutputs() const override;

    [[nodiscard]] const std::vector<nvinfer1::Dims>& inputDims() const override;
    [[nodiscard]] const std::vector<nvinfer1::Dims>& outputDims() const override;
    [[nodiscard]] const std::vector<size_t>& inputSizes() const override;
    [[nodiscard]] const std::vector<size_t>& outputSizes() const override;

    float* outputHost(size_t i) override;
    [[nodiscard]] const float* outputHost(size_t i) const override;

    bool enqueue(const cv::Mat& nchwBlob) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace yolo26

#endif
