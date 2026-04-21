#include <fstream>
#include <filesystem>
#include <iostream>
#include "NvOnnxParser.h"
#include "TrtEngineCache.h"
#include "Yolo26Seg.h"

namespace fs = std::filesystem;

cv::Mat sigmoid(const cv::Mat& inputMat) {
    cv::Mat outputMat;
    cv::exp(-inputMat, outputMat);
    cv::divide(1, 1 + outputMat, outputMat);
    return outputMat;
}

Yolo26Seg::Yolo26Seg(const Config& config) {
    modelFile_ = config.modelFile;
    scoreThreshold_ = config.scoreThreshold;
    logger_ = new Logger();
}

Yolo26Seg::~Yolo26Seg() {
    if (nullptr != context_) {
        delete context_;
        context_ = nullptr;
    }
    if (nullptr != engine_) {
        delete engine_;
        engine_ = nullptr;
    }
}

bool Yolo26Seg::init() {

    bool onnxFound = fs::exists(fs::absolute(modelFile_));
    if (!onnxFound) {
        std::cerr << "Cannot find model file: " << modelFile_ << std::endl;
        return false;
    }

    std::cout << "Try loading onnx file: " << modelFile_ << std::endl;
    bool onnxInitFlag = initFromOnnx(fs::absolute(modelFile_).string());
    if (onnxInitFlag) {
        std::cout << "Loading succeed..." << std::endl;
    } else {
        std::cerr << "Loading failed... " << std::endl;
    }

    return onnxInitFlag;
}

std::vector<YOLOInferResult> Yolo26Seg::inference(const cv::Mat& image) {

    imageWidth_ = image.cols;
    imageHeight_ = image.rows;

    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(modelWidth_, modelHeight_), cv::Scalar(), true,
                                          false, CV_32F);

    cudaSetDevice(deviceId_);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void** outputHostBuffers = new void*[outputsNum_];
    for (size_t i = 0; i < outputsNum_; i++) {
        cudaHostAlloc((void**)&outputHostBuffers[i], outputSizes_[i], 0);
    }
    void** inputDeviceBuffers = new void*[inputsNum_];
    for (size_t i = 0; i < inputsNum_; i++) {
        cudaMalloc(&inputDeviceBuffers[i], inputSizes_[i]);
    }
    void** outputDeviceBuffers = new void*[outputsNum_];
    for (size_t i = 0; i < outputsNum_; i++) {
        cudaMalloc(&outputDeviceBuffers[i], outputSizes_[i]);
    }
    for (size_t i = 0; i < inputsNum_; i++) {
        cudaMemcpyAsync(inputDeviceBuffers[i], blob.data, inputSizes_[i], cudaMemcpyHostToDevice, stream);
    }
    for (size_t i = 0; i < inputsNum_; i++) {
        context_->setInputTensorAddress(vecInputLayerNames_[i].c_str(), inputDeviceBuffers[i]);
    }
    for (size_t i = 0; i < outputsNum_; i++) {
        context_->setOutputTensorAddress(vecOutputLayerNames_[i].c_str(), outputDeviceBuffers[i]);
    }

#if NV_TENSORRT_MAJOR >= 10
    context_->enqueueV3(stream);
#else
    context_->enqueueV2(deviceBuffers, stream, nullptr);
#endif

    for (size_t i = 0; i < outputsNum_; i++) {
        cudaMemcpyAsync(outputHostBuffers[i], outputDeviceBuffers[i], outputSizes_[i], cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::vector<InferenceOutput> inferOutputs{};

    for (size_t i = 0; i < outputsNum_; i++) {
        InferenceOutput out;
        out.data = std::shared_ptr<float>(static_cast<float*>(outputHostBuffers[i]), [](float* p) { cudaFreeHost(p); });
        out.size = outputSizes_[i];
        inferOutputs.push_back(out);
    }
    for (size_t i = 0; i < inputsNum_; i++) {
        cudaFree(inputDeviceBuffers[i]);
    }
    for (size_t i = 0; i < outputsNum_; i++) {
        cudaFree(outputDeviceBuffers[i]);
    }

    return postProcessing(inferOutputs);
}

bool Yolo26Seg::initFromOnnx(const std::string& onnxPath) {

    const std::string enginePath = trt_engine_cache::enginePathFromOnnx(onnxPath);

    if (!trt_engine_cache::shouldRebuildEngine(onnxPath, enginePath)) {
        std::cout << "Loading TensorRT engine cache: " << enginePath << std::endl;
        if (trt_engine_cache::deserializeEngine(enginePath, *logger_, engine_, context_)) {
            retrieveNetInfo();
            if (inputsNum_ != inputSizes_.size() || outputsNum_ != outputSizes_.size()) {
                std::cerr << "Error network's input/output number..." << std::endl;
                return false;
            }
            modelHeight_ = vecInputDims_[0].d[2];
            modelWidth_ = vecInputDims_[0].d[3];
            return true;
        }
        std::cerr << "Engine cache load failed, rebuilding from ONNX..." << std::endl;
        if (engine_ != nullptr) {
            delete engine_;
            engine_ = nullptr;
        }
        if (context_ != nullptr) {
            delete context_;
            context_ = nullptr;
        }
    }

    std::ifstream onnxFilestream(onnxPath, std::ios::binary);
    if (!onnxFilestream.is_open()) {
        std::cerr << "Open onnx file failed: " << onnxPath;
        return false;
    }

    onnxFilestream.seekg(0, std::ios::end);
    size_t onnxSize = onnxFilestream.tellg();
    onnxFilestream.seekg(0, std::ios::beg);

    std::vector<char> onnxData(onnxSize);
    onnxFilestream.read(onnxData.data(), onnxSize);
    onnxFilestream.close();

    nvinfer1::IBuilder* iBuilder = nvinfer1::createInferBuilder(*logger_);
    nvinfer1::NetworkDefinitionCreationFlags flags{
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};
    nvinfer1::INetworkDefinition* network = iBuilder->createNetworkV2(flags);
    nvinfer1::IBuilderConfig* config = iBuilder->createBuilderConfig();
    nvonnxparser::IParser* onnxParser = nvonnxparser::createParser(*network, *logger_);
    if (!onnxParser->parse(onnxData.data(), onnxSize)) {
        std::cerr << "Parse onnx buffer failed.." << std::endl;
        delete onnxParser;
        delete network;
        delete config;
        delete iBuilder;
        return false;
    }

    std::cout << "Building inference environment, may take very long....(first build or ONNX newer than .engine)"
              << std::endl;
    engine_ = iBuilder->buildEngineWithConfig(*network, *config);
    if (nullptr == engine_) {
        std::cerr << "TRT engine create failed.." << std::endl;
        delete onnxParser;
        delete network;
        delete config;
        delete iBuilder;
        return false;
    }
    context_ = engine_->createExecutionContext();
    if (nullptr == context_) {
        std::cerr << "TRT context create failed.." << std::endl;
        delete onnxParser;
        delete network;
        delete config;
        delete iBuilder;
        return false;
    }
    std::cout << "Building environment finished" << std::endl;

    if (trt_engine_cache::serializeEngineToFile(engine_, enginePath)) {
        std::cout << "Saved TensorRT engine cache: " << enginePath << std::endl;
    } else {
        std::cerr << "Warning: could not save .engine cache (next run will rebuild)" << std::endl;
    }

    retrieveNetInfo();

    delete onnxParser;
    delete network;
    delete config;
    delete iBuilder;

    if (inputsNum_ != inputSizes_.size() || outputsNum_ != outputSizes_.size()) {
        std::cerr << "Error network's input/output number..." << std::endl;
        return false;
    }

    modelHeight_ = vecInputDims_[0].d[2];
    modelWidth_ = vecInputDims_[0].d[3];

    return true;
}

void Yolo26Seg::retrieveNetInfo() {
    int ioNumbers = engine_->getNbIOTensors();
    std::cout << "number of io layers: " << ioNumbers << std::endl;

    for (int i = 0; i < ioNumbers; i++) {
        const char* layerName = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode type = engine_->getTensorIOMode(layerName);
        nvinfer1::Dims dim = engine_->getTensorShape(layerName);

        if (nvinfer1::TensorIOMode::kINPUT == type) {
            vecInputDims_.push_back(dim);
            vecInputLayerNames_.emplace_back(layerName);
            std::cout << "input layer: " << layerName << std::endl;
            size_t bufferSize = 1 * sizeof(float);
            for (int j = 0; j < dim.nbDims; j++) {
                std::cout << "\t dim" << j << " size: " << dim.d[j] << std::endl;
                bufferSize *= dim.d[j];
            }
            inputSizes_.push_back(bufferSize);
        } else if (nvinfer1::TensorIOMode::kOUTPUT == type) {
            vecOutputDims_.push_back(dim);
            vecOutputLayerNames_.emplace_back(layerName);
            std::cout << "output layer: " << layerName << std::endl;
            size_t bufferSize = 1 * sizeof(float);
            for (int j = 0; j < dim.nbDims; j++) {
                std::cout << "\t dim" << j << " size: " << dim.d[j] << std::endl;
                bufferSize *= dim.d[j];
            }
            outputSizes_.push_back(bufferSize);
        }
    }
    inputsNum_ = vecInputDims_.size();
    outputsNum_ = vecOutputDims_.size();
}

std::vector<YOLOInferResult> Yolo26Seg::postProcessing(std::vector<InferenceOutput>& inferOutputs) const {
    if (outputsNum_ != 2 || inferOutputs.size() != 2) {
        std::cout << "Yolo10/26 segmentation model can only have two outputs" << std::endl;
        return {};
    }

    // det 是 3 维、proto 是 4 维，tensor 顺序可能互换
    int detIndex, protoIndex;

    if (vecOutputDims_[0].nbDims == 3) {
        detIndex = 0, protoIndex = 1;
    } else {
        detIndex = 1, protoIndex = 0;
    }

    size_t numDetection = vecOutputDims_[detIndex].d[1];
    size_t dimension = vecOutputDims_[detIndex].d[2];
    size_t numProtos = vecOutputDims_[protoIndex].d[1];
    size_t protoH = vecOutputDims_[protoIndex].d[2];
    size_t protoW = vecOutputDims_[protoIndex].d[3];

    float scaleX = static_cast<float>(protoW) / static_cast<float>(vecInputDims_[0].d[3]);
    float scaleY = static_cast<float>(protoH) / static_cast<float>(vecInputDims_[0].d[2]);

    if (dimension < 38 || numProtos != 32) {
        return {};
    }

    auto* detBuff = static_cast<float*>(inferOutputs[detIndex].data.get());
    auto* protoBuff = static_cast<float*>(inferOutputs[protoIndex].data.get());

    cv::Mat output0Mat(static_cast<int>(numDetection), static_cast<int>(dimension), CV_32F, detBuff);
    cv::Mat output1Mat(static_cast<int>(numProtos), static_cast<int>(protoH * protoW), CV_32F, protoBuff);

    std::vector<YOLOInferResult> inferResults;

    for (int i = 0; i < numDetection; i++) {
        auto* data = output0Mat.ptr<float>(i);

        float x1 = data[0], y1 = data[1], x2 = data[2], y2 = data[3], score = data[4];
        int classId = static_cast<int>(data[5]);
        float w = x2 - x1, h = y2 - y1;
        if (score < scoreThreshold_)
            continue;

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

        cv::Mat maskMapNetSize = cv::Mat::zeros(cv::Size(modelWidth_, modelHeight_), CV_8UC1);

        cv::Rect tempRect = cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                     cv::Point(static_cast<int>(x2), static_cast<int>(y2))) &
            cv::Rect(0, 0, modelWidth_, modelHeight_);
        if (objMaskROIThreshold.size() != tempRect.size()) {
            cv::resize(objMaskROIThreshold, objMaskROIThreshold, tempRect.size(), cv::INTER_NEAREST);
        }
        objMaskROIThreshold.copyTo(maskMapNetSize(tempRect));

        cv::Mat maskMapImageSize;
        cv::resize(maskMapNetSize, maskMapImageSize, cv::Size(imageWidth_, imageHeight_), 0, 0, cv::INTER_NEAREST);

        YOLOInferResult result;
        result.rect =
            cv::Rect(
                cv::Point(static_cast<int>(x1 / static_cast<float>(modelWidth_) * static_cast<float>(imageWidth_)),
                          static_cast<int>(y1 / static_cast<float>(modelHeight_) * static_cast<float>(imageHeight_))),
                cv::Point(static_cast<int>(x2 / static_cast<float>(modelWidth_) * static_cast<float>(imageWidth_)),
                          static_cast<int>(y2 / static_cast<float>(modelHeight_) * static_cast<float>(imageHeight_)))) &
            cv::Rect(0, 0, imageWidth_, imageHeight_);
        result.mask = maskMapImageSize;
        result.score = score;
        result.classIndex = classId;
        result.index = i;

        inferResults.push_back(result);
    }
    return inferResults;
}
