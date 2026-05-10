#ifndef YOLO26_YOLO26OBB_H
#define YOLO26_YOLO26OBB_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo26/Yolo26VariantBackend.h"

struct ObbInferResult {
    cv::RotatedRect rrect;
    float score = 0.f;
    size_t classIndex = 0;
    size_t index = 0;
};

struct ObbConfig {
    std::string modelFile;
    float scoreThreshold = 0.45f;
    size_t maxDetections = 50;
    size_t numClasses = 80;
    bool end2endLayout = true;
    bool angleInRadians = true;
    yolo26::Yolo26BackendKind backendKind = yolo26::Yolo26BackendKind::kTensorRT;
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
    yolo26::Yolo26BackendKind backendKind_;

    int32_t imageWidth_ = 0;
    int32_t imageHeight_ = 0;

    float letterboxScale_ = 1.f;
    float letterboxPadW_ = 0.f;
    float letterboxPadH_ = 0.f;

    std::unique_ptr<yolo26::Yolo26VariantBackend> backend_;

    std::vector<ObbInferResult> postProcessing(float* output0) const;
};

#endif
