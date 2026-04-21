#ifndef YOLO26_CPP_YOLO26DETECT_H
#define YOLO26_CPP_YOLO26DETECT_H

// YOLO26 检测：nc=80、end2end。输入 1×3×H×W（尺寸看引擎），输出 1×N×6：xyxy、conf、class，无 NMS

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

struct DetectInferResult {
    cv::Rect rect;
    float score = 0.f;
    size_t classIndex = 0;
    size_t index = 0;
};

struct DetectConfig {
    std::string modelFile;
    float scoreThreshold = 0.45f; // 置信度下限
    size_t maxDetections = 50;    // 最多保留几条，0 不限
    size_t numClasses = 80;       // 和 yaml 里 nc 一致
};

class Yolo26DetectLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity < Severity::kINFO) {
            std::cout << message << std::endl;
        }
    }
};

class Yolo26Detect {
public:
    explicit Yolo26Detect(const DetectConfig& config);
    ~Yolo26Detect();

    Yolo26Detect(const Yolo26Detect&) = delete;
    Yolo26Detect& operator=(const Yolo26Detect&) = delete;

    bool init();

    std::vector<DetectInferResult> inference(const cv::Mat& image);

private:
    std::string modelFile_;
    float scoreThreshold_ = 0.45f;
    size_t maxDetections_ = 50;
    size_t numClasses_ = 80;

    int32_t deviceId_ = 0;
    int32_t imageWidth_ = 0;
    int32_t imageHeight_ = 0;
    int32_t modelWidth_ = 0;
    int32_t modelHeight_ = 0;
    // letterbox 反算回原图
    float letterboxScale_ = 1.f;
    float letterboxPadW_ = 0.f;
    float letterboxPadH_ = 0.f;

    cudaStream_t stream_ = nullptr;
    std::vector<void*> inputDev_;
    std::vector<void*> outputDev_;
    std::vector<void*> outputHostPinned_;

    size_t inputsNum_ = 0;
    std::vector<nvinfer1::Dims> vecInputDims_;
    std::vector<std::string> vecInputLayerNames_;
    std::vector<size_t> inputSizes_;

    size_t outputsNum_ = 0;
    std::vector<nvinfer1::Dims> vecOutputDims_;
    std::vector<std::string> vecOutputLayerNames_;
    std::vector<size_t> outputSizes_;

    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    Yolo26DetectLogger* logger_ = nullptr;

    bool initFromOnnx(const std::string& onnxPath);
    void retrieveNetInfo();
    bool allocateBuffers();
    void freeBuffers();
    std::vector<DetectInferResult> postProcessing(float* output0) const;
};

#endif
