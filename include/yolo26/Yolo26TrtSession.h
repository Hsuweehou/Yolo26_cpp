#ifndef YOLO26_YOLO26TRTSESSION_H
#define YOLO26_YOLO26TRTSESSION_H

// TensorRT 10+ 实现 Yolo26VariantBackend

#include <iostream>

#include <cuda_runtime.h>

#include "yolo26/Yolo26VariantBackend.h"

namespace yolo26 {

class Yolo26TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity < Severity::kINFO) {
            std::cout << message << std::endl;
        }
    }
};

class Yolo26TrtSession : public Yolo26VariantBackend {
public:
    Yolo26TrtSession();
    ~Yolo26TrtSession() override;

    Yolo26BackendKind kind() const noexcept override { return Yolo26BackendKind::kTensorRT; }

    bool loadFromOnnx(const std::string& onnxPath, std::string_view taskTag) override;

    [[nodiscard]] bool ready() const override {
        return engine_ != nullptr && context_ != nullptr && stream_ != nullptr;
    }

    [[nodiscard]] int32_t deviceId() const override { return deviceId_; }
    [[nodiscard]] int modelWidth() const override { return modelW_; }
    [[nodiscard]] int modelHeight() const override { return modelH_; }

    [[nodiscard]] size_t numInputs() const override { return inputsNum_; }
    [[nodiscard]] size_t numOutputs() const override { return outputsNum_; }

    [[nodiscard]] const std::vector<nvinfer1::Dims>& inputDims() const override { return vecInputDims_; }
    [[nodiscard]] const std::vector<nvinfer1::Dims>& outputDims() const override { return vecOutputDims_; }
    [[nodiscard]] const std::vector<size_t>& inputSizes() const override { return inputSizes_; }
    [[nodiscard]] const std::vector<size_t>& outputSizes() const override { return outputSizes_; }

    float* outputHost(size_t i) override;
    [[nodiscard]] const float* outputHost(size_t i) const override;

    bool enqueue(const cv::Mat& nchwBlob) override;

    nvinfer1::ICudaEngine* engine() { return engine_; }
    nvinfer1::IExecutionContext* context() { return context_; }

private:
    void releaseEngine();
    void retrieveNetInfo();
    bool allocateBuffers();
    void freeBuffers();

    Yolo26TrtLogger logger_;
    int32_t deviceId_ = 0;
    int modelW_ = 0;
    int modelH_ = 0;

    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    size_t inputsNum_ = 0;
    size_t outputsNum_ = 0;
    std::vector<nvinfer1::Dims> vecInputDims_;
    std::vector<nvinfer1::Dims> vecOutputDims_;
    std::vector<std::string> vecInputLayerNames_;
    std::vector<std::string> vecOutputLayerNames_;
    std::vector<size_t> inputSizes_;
    std::vector<size_t> outputSizes_;

    cudaStream_t stream_ = nullptr;
    std::vector<void*> inputDev_;
    std::vector<void*> outputDev_;
    std::vector<void*> outputHostPinned_;
};

} // namespace yolo26

#endif
