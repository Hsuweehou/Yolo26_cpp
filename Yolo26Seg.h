#ifndef YOLO26_CPP_YOLO26SEG_H
#define YOLO26_CPP_YOLO26SEG_H

// 实例分割：输出一般是 det + mask 原型两路

#include <opencv2/opencv.hpp>
#include <string>
#include "NvInfer.h"

struct InferenceOutput {
    std::shared_ptr<void> data = nullptr;
    uint32_t size;
};

typedef struct YOLOInferResult {
    cv::Rect rect;
    cv::Mat mask;
    float score;
    size_t classIndex;
    size_t index; // 输出里的第几条
} YOLOInferResult;


typedef struct Config {
    std::string modelFile;
    float scoreThreshold = 0.25f;
} Config;

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity < Severity::kINFO) {
            std::cout << message << std::endl;
        }
    }
};

class Yolo26Seg {
public:
    explicit Yolo26Seg(const Config& config);

    ~Yolo26Seg();

    bool init();

    std::vector<YOLOInferResult> inference(const cv::Mat& image);

private:
    std::string modelFile_;
    float scoreThreshold_;

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
    Logger* logger_ = nullptr;

    bool initFromOnnx(const std::string& onnxPath);
    void retrieveNetInfo();
    std::vector<YOLOInferResult> postProcessing(std::vector<InferenceOutput>& inferOutputs) const;

};


#endif // YOLO26_CPP_YOLO26SEG_H
