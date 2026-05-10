#ifndef YOLO26_YOLO26SEG_H
#define YOLO26_YOLO26SEG_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo26/Yolo26VariantBackend.h"

typedef struct YOLOInferResult {
    cv::Rect rect;
    cv::Mat mask;
    float score;
    size_t classIndex;
    size_t index;
} YOLOInferResult;

typedef struct Config {
    std::string modelFile;
    float scoreThreshold = 0.25f;
    yolo26::Yolo26BackendKind backendKind = yolo26::Yolo26BackendKind::kTensorRT;
} Config;

class Yolo26Seg {
public:
    explicit Yolo26Seg(const Config& config);

    ~Yolo26Seg();

    bool init();

    std::vector<YOLOInferResult> inference(const cv::Mat& image);

private:
    std::string modelFile_;
    float scoreThreshold_;
    yolo26::Yolo26BackendKind backendKind_;

    int32_t imageWidth_ = 0;
    int32_t imageHeight_ = 0;

    float letterboxScale_ = 1.f;
    float letterboxPadW_ = 0.f;
    float letterboxPadH_ = 0.f;

    std::unique_ptr<yolo26::Yolo26VariantBackend> backend_;

    std::vector<YOLOInferResult> postProcessing(int detIndex, int protoIndex);
};

#endif
