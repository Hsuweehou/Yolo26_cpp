#ifndef YOLO26_YOLO26POSE_H
#define YOLO26_YOLO26POSE_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo26/Yolo26VariantBackend.h"

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
    float nmsThreshold = 0.5f;
    size_t numKeypoints = 17;
    size_t numClasses = 80;
    bool end2endLayout = true;
    size_t maxDetections = 300;
    yolo26::Yolo26BackendKind backendKind = yolo26::Yolo26BackendKind::kTensorRT;
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
    yolo26::Yolo26BackendKind backendKind_;

    int32_t imageWidth_ = 0;
    int32_t imageHeight_ = 0;

    float letterboxScale_ = 1.f;
    float letterboxPadW_ = 0.f;
    float letterboxPadH_ = 0.f;

    std::unique_ptr<yolo26::Yolo26VariantBackend> backend_;

    std::vector<PoseInferResult> postProcessing(float* output0) const;
};

#endif
