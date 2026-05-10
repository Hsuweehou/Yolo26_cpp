#ifndef YOLO26_YOLO26VARIANTBACKEND_H
#define YOLO26_YOLO26VARIANTBACKEND_H

// Pure virtual inference backend (cf. YoloDetBackendImpl in YOLO26_TensorRT_ONNX): mock for tests, or ONNXRuntime / OpenVINO later.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

namespace yolo26 {

/// Backend kind：TensorRT（GPU）与可选 ONNX Runtime（CPU，见 CMake ONNXRUNTIME）。
enum class Yolo26BackendKind {
    kTensorRT = 0,
    kOnnxRuntime = 1,
    kOpenVINO = 2,
};

/// Shared interface: load ONNX/engine once, then enqueue NCHW blob.
class Yolo26VariantBackend {
public:
    virtual ~Yolo26VariantBackend() = default;

    Yolo26VariantBackend(const Yolo26VariantBackend&) = delete;
    Yolo26VariantBackend& operator=(const Yolo26VariantBackend&) = delete;

    virtual Yolo26BackendKind kind() const noexcept = 0;

    virtual bool loadFromOnnx(const std::string& onnxPath, std::string_view taskTag) = 0;
    [[nodiscard]] virtual bool ready() const = 0;

    [[nodiscard]] virtual int32_t deviceId() const = 0;
    [[nodiscard]] virtual int modelWidth() const = 0;
    [[nodiscard]] virtual int modelHeight() const = 0;

    [[nodiscard]] virtual size_t numInputs() const = 0;
    [[nodiscard]] virtual size_t numOutputs() const = 0;

    [[nodiscard]] virtual const std::vector<nvinfer1::Dims>& inputDims() const = 0;
    [[nodiscard]] virtual const std::vector<nvinfer1::Dims>& outputDims() const = 0;
    [[nodiscard]] virtual const std::vector<size_t>& inputSizes() const = 0;
    [[nodiscard]] virtual const std::vector<size_t>& outputSizes() const = 0;

    virtual float* outputHost(size_t i) = 0;
    [[nodiscard]] virtual const float* outputHost(size_t i) const = 0;

    /// 单输入 NCHW CV_32F blob
    virtual bool enqueue(const cv::Mat& nchwBlob) = 0;

protected:
    Yolo26VariantBackend() = default;
};

} // namespace yolo26

#endif
