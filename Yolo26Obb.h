#ifndef YOLO26_CPP_YOLO26OBB_H
#define YOLO26_CPP_YOLO26OBB_H

// 旋转框：端到端每行 7 个数，中心+wh+得分+类+角，不是 detect 那种 xyxy

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

struct ObbInferenceOutput {
    std::shared_ptr<void> data = nullptr;
    uint32_t size = 0;
};

struct ObbInferResult {
    cv::RotatedRect rrect; // 原图坐标，角度单位度（OpenCV）
    float score = 0.f;
    size_t classIndex = 0;
    size_t index = 0;
};

struct ObbConfig {
    std::string modelFile;
    float scoreThreshold = 0.45f;
    size_t maxDetections = 50;
    size_t numClasses = 80;
    bool end2endLayout = true; // false：5 维框 + 每类得分
    bool angleInRadians = true; // 网络输出角是弧度还是度
};

class Yolo26ObbLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity < Severity::kINFO) {
            std::cout << message << std::endl;
        }
    }
};

class Yolo26Obb {
public:
    explicit Yolo26Obb(const ObbConfig& config);
    ~Yolo26Obb();

    Yolo26Obb(const Yolo26Obb&) = delete;
    Yolo26Obb& operator=(const Yolo26Obb&) = delete;

    bool init();

    std::vector<ObbInferResult> inference(const cv::Mat& image);

private:
    std::string modelFile_;
    float scoreThreshold_ = 0.45f;
    size_t maxDetections_ = 50;
    size_t numClasses_ = 80;
    bool end2endLayout_ = true;
    bool angleInRadians_ = true;

    int32_t deviceId_ = 0;
    int32_t imageWidth_ = 0;
    int32_t imageHeight_ = 0;
    int32_t modelWidth_ = 0;
    int32_t modelHeight_ = 0;
    // letterbox 反算回原图
    float letterboxScale_ = 1.f;
    float letterboxPadW_ = 0.f;
    float letterboxPadH_ = 0.f;

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
    Yolo26ObbLogger* logger_ = nullptr;

    bool initFromOnnx(const std::string& onnxPath);
    void retrieveNetInfo();
    std::vector<ObbInferResult> postProcessing(std::vector<ObbInferenceOutput>& inferOutputs) const;
};

#endif
