#ifndef YOLO26_YOLO26DETECT_H
#define YOLO26_YOLO26DETECT_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo26/Yolo26VariantBackend.h"

struct DetectInferResult {
    cv::Rect rect;
    float score = 0.f;
    size_t classIndex = 0;
    size_t index = 0;
};

struct DetectConfig {
    std::string modelFile;
    float scoreThreshold = 0.45f;
    size_t maxDetections = 50;
    size_t numClasses = 80;
    yolo26::Yolo26BackendKind backendKind = yolo26::Yolo26BackendKind::kTensorRT;
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
    yolo26::Yolo26BackendKind backendKind_;

    int32_t imageWidth_ = 0;
    int32_t imageHeight_ = 0;

    float letterboxScale_ = 1.f;
    float letterboxPadW_ = 0.f;
    float letterboxPadH_ = 0.f;

    std::unique_ptr<yolo26::Yolo26VariantBackend> backend_;

    std::vector<DetectInferResult> postProcessing(float* output0) const;
};

#endif
