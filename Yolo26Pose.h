#ifndef YOLO26_CPP_YOLO26POSE_H
#define YOLO26_CPP_YOLO26POSE_H

// 姿态：默认 nc=80、17 点、end2end。每行 C=6+K*3：前 6 维同检测，后面是关键点 (x,y,vis)
// end2end=false 时是旧格式 4+nc+K*3，要走 NMS

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

struct PoseInferenceOutput {
    std::shared_ptr<void> data = nullptr;
    uint32_t size = 0;
};

// 已换算到原图像素
struct PoseKeypoint {
    float x = 0.f;
    float y = 0.f;
    float conf = 0.f;
};

struct PoseInferResult {
    cv::Rect rect;
    std::vector<PoseKeypoint> keypoints;
    float score = 0.f;
    size_t classIndex = 0;
    size_t index = 0;
};

struct PoseConfig {
    std::string modelFile;
    float scoreThreshold = 0.25f;
    float nmsThreshold = 0.5f; // 仅 end2end=false 时用
    size_t numKeypoints = 17;
    size_t numClasses = 80;
    bool end2endLayout = true; // true：一行 C=6+K*3，无 NMS
    size_t maxDetections = 300;  // 0 不截断
};

class Yolo26PoseLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity < Severity::kINFO) {
            std::cout << message << std::endl;
        }
    }
};

class Yolo26Pose {
public:
    explicit Yolo26Pose(const PoseConfig& config);
    ~Yolo26Pose();

    Yolo26Pose(const Yolo26Pose&) = delete;
    Yolo26Pose& operator=(const Yolo26Pose&) = delete;

    bool init();

    std::vector<PoseInferResult> inference(const cv::Mat& image);

private:
    std::string modelFile_;
    float scoreThreshold_ = 0.25f;
    float nmsThreshold_ = 0.5f;
    size_t numKeypoints_ = 17;
    size_t numClasses_ = 80;
    bool end2endLayout_ = true;
    size_t maxDetections_ = 300;

    int32_t deviceId_ = 0;
    int32_t imageWidth_ = 0;
    int32_t imageHeight_ = 0;
    int32_t modelWidth_ = 0;
    int32_t modelHeight_ = 0;

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
    Yolo26PoseLogger* logger_ = nullptr;

    bool initFromOnnx(const std::string& onnxPath);
    void retrieveNetInfo();
    std::vector<PoseInferResult> postProcessing(std::vector<PoseInferenceOutput>& inferOutputs) const;
};

#endif
