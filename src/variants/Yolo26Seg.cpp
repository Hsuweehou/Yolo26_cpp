#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "yolo26/Yolo26BackendFactory.h"
#include "yolo26/Yolo26Letterbox.h"
#include "yolo26/Yolo26Seg.h"

namespace fs = std::filesystem;

namespace {

cv::Mat sigmoid(const cv::Mat& inputMat) {
    cv::Mat outputMat;
    cv::exp(-inputMat, outputMat);
    cv::divide(1, 1 + outputMat, outputMat);
    return outputMat;
}

} // namespace

Yolo26Seg::Yolo26Seg(const Config& config) {
    modelFile_ = config.modelFile;
    scoreThreshold_ = config.scoreThreshold;
    backendKind_ = config.backendKind;
}

Yolo26Seg::~Yolo26Seg() = default;

bool Yolo26Seg::init() {
    if (!fs::exists(fs::absolute(modelFile_))) {
        std::cerr << "Cannot find model file: " << modelFile_ << std::endl;
        return false;
    }

    backend_ = yolo26::CreateYolo26VariantBackend(backendKind_);
    if (!backend_) {
        std::cerr << "Yolo26Seg: failed to create backend\n";
        return false;
    }

    std::cout << "Try loading onnx file (seg): " << modelFile_ << std::endl;
    const bool ok = backend_->loadFromOnnx(fs::absolute(modelFile_).string(), "seg");
    if (ok) {
        std::cout << "Loading succeed (seg)..." << std::endl;
    } else {
        std::cerr << "Loading failed (seg)..." << std::endl;
    }
    return ok;
}

std::vector<YOLOInferResult> Yolo26Seg::inference(const cv::Mat& image) {
    imageWidth_ = image.cols;
    imageHeight_ = image.rows;

    cv::Mat letterboxed;
    letterboxUltralytics(image, backend_->modelWidth(), backend_->modelHeight(), letterboxed, letterboxScale_,
                         letterboxPadW_, letterboxPadH_);

    cv::Mat blob =
        cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(backend_->modelWidth(), backend_->modelHeight()),
                                 cv::Scalar(), true, false, CV_32F);

    if (!backend_->enqueue(blob)) {
        return {};
    }

    if (backend_->numOutputs() != 2) {
        std::cout << "Yolo26Seg: segmentation model expects two outputs, got " << backend_->numOutputs() << std::endl;
        return {};
    }

    int detIndex = 0;
    int protoIndex = 1;
    if (backend_->outputDims()[0].nbDims != 3) {
        detIndex = 1;
        protoIndex = 0;
    }

    return postProcessing(detIndex, protoIndex);
}

std::vector<YOLOInferResult> Yolo26Seg::postProcessing(int detIndex, int protoIndex) {
    if (!backend_ || backend_->numOutputs() != 2) {
        std::cout << "Yolo26Seg: can only have two outputs" << std::endl;
        return {};
    }

    const nvinfer1::Dims& detDim = backend_->outputDims()[static_cast<size_t>(detIndex)];
    const nvinfer1::Dims& protoDim = backend_->outputDims()[static_cast<size_t>(protoIndex)];

    size_t numDetection = static_cast<size_t>(detDim.d[1]);
    size_t dimension = static_cast<size_t>(detDim.d[2]);
    size_t numProtos = static_cast<size_t>(protoDim.d[1]);
    size_t protoH = static_cast<size_t>(protoDim.d[2]);
    size_t protoW = static_cast<size_t>(protoDim.d[3]);

    const int mw = backend_->modelWidth();
    const int mh = backend_->modelHeight();
    const int inW = backend_->inputDims()[0].d[3];
    const int inH = backend_->inputDims()[0].d[2];

    float scaleX = static_cast<float>(protoW) / static_cast<float>(inW);
    float scaleY = static_cast<float>(protoH) / static_cast<float>(inH);

    if (dimension < 38 || numProtos != 32) {
        return {};
    }

    auto* detBuff = backend_->outputHost(static_cast<size_t>(detIndex));
    auto* protoBuff = backend_->outputHost(static_cast<size_t>(protoIndex));
    if (detBuff == nullptr || protoBuff == nullptr) {
        return {};
    }

    cv::Mat output0Mat(static_cast<int>(numDetection), static_cast<int>(dimension), CV_32F, detBuff);
    cv::Mat output1Mat(static_cast<int>(numProtos), static_cast<int>(protoH * protoW), CV_32F, protoBuff);

    std::vector<YOLOInferResult> inferResults;

    const float sx = letterboxScale_;
    const float px = letterboxPadW_;
    const float py = letterboxPadH_;
    const int contentLeft = static_cast<int>(std::floor(px));
    const int contentTop = static_cast<int>(std::floor(py));
    const int contentW = static_cast<int>(std::round(static_cast<float>(imageWidth_) * sx));
    const int contentH = static_cast<int>(std::round(static_cast<float>(imageHeight_) * sx));

    for (int i = 0; i < static_cast<int>(numDetection); i++) {
        auto* data = output0Mat.ptr<float>(i);

        float x1 = data[0], y1 = data[1], x2 = data[2], y2 = data[3], score = data[4];
        int classId = static_cast<int>(data[5]);
        float w = x2 - x1, h = y2 - y1;
        if (score < scoreThreshold_) {
            continue;
        }

        float protoMaskParams[32];
        std::copy(data + 6, data + 38, protoMaskParams);

        cv::Mat protoMaskParamsMat(1, 32, CV_32F, protoMaskParams);
        cv::Mat weightedProtoMask;
        cv::gemm(protoMaskParamsMat, output1Mat, 1.0, cv::Mat(), 0, weightedProtoMask);

        cv::Mat reshapedMat = weightedProtoMask.reshape(1, static_cast<int>(protoH));
        cv::Mat sigmoidMat = sigmoid(reshapedMat);

        cv::Rect objMaskRectResized =
            cv::Rect(cv::Point(static_cast<int>(x1 * scaleX), static_cast<int>(y1 * scaleY)),
                     cv::Point(static_cast<int>(x2 * scaleX), static_cast<int>(y2 * scaleY))) &
            cv::Rect(0, 0, static_cast<int>(protoW), static_cast<int>(protoH));
        cv::Mat objMaskROIResized = sigmoidMat(objMaskRectResized).clone();

        cv::Mat objMaskROI;
        cv::resize(objMaskROIResized, objMaskROI, cv::Size(static_cast<int>(w), static_cast<int>(h)), cv::INTER_CUBIC);

        cv::Size blurKernel(3, 3);
        cv::Mat objMaskROIBlurred;
        cv::blur(objMaskROI, objMaskROIBlurred, blurKernel);

        cv::Mat objMaskROIThreshold;
        cv::threshold(objMaskROIBlurred, objMaskROIThreshold, 0.5, 255, cv::THRESH_BINARY);
        objMaskROIThreshold.convertTo(objMaskROIThreshold, CV_8UC1);

        cv::Mat maskMapNetSize = cv::Mat::zeros(cv::Size(mw, mh), CV_8UC1);

        cv::Rect tempRect = cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                     cv::Point(static_cast<int>(x2), static_cast<int>(y2))) &
            cv::Rect(0, 0, mw, mh);
        if (objMaskROIThreshold.size() != tempRect.size()) {
            cv::resize(objMaskROIThreshold, objMaskROIThreshold, tempRect.size(), cv::INTER_NEAREST);
        }
        objMaskROIThreshold.copyTo(maskMapNetSize(tempRect));

        cv::Mat maskMapImageSize;
        cv::Rect roiNet(contentLeft, contentTop, contentW, contentH);
        roiNet &= cv::Rect(0, 0, mw, mh);
        if (roiNet.width > 0 && roiNet.height > 0) {
            cv::Mat cropped = maskMapNetSize(roiNet);
            cv::resize(cropped, maskMapImageSize, cv::Size(imageWidth_, imageHeight_), 0, 0, cv::INTER_NEAREST);
        } else {
            maskMapImageSize = cv::Mat::zeros(cv::Size(imageWidth_, imageHeight_), CV_8UC1);
        }

        YOLOInferResult result;
        const float x1o = (x1 - px) / sx;
        const float y1o = (y1 - py) / sx;
        const float x2o = (x2 - px) / sx;
        const float y2o = (y2 - py) / sx;
        result.rect = cv::Rect(cv::Point(static_cast<int>(std::floor(x1o)), static_cast<int>(std::floor(y1o))),
                                 cv::Point(static_cast<int>(std::ceil(x2o)), static_cast<int>(std::ceil(y2o)))) &
            cv::Rect(0, 0, imageWidth_, imageHeight_);
        result.mask = maskMapImageSize;
        result.score = score;
        result.classIndex = static_cast<size_t>(classId);
        result.index = static_cast<size_t>(i);

        inferResults.push_back(result);
    }
    return inferResults;
}
