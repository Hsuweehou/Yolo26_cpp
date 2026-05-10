#include "yolo26/Yolo26TrtSession.h"

#include <fstream>
#include <iostream>

#include <NvOnnxParser.h>

#include "yolo26/TrtEngineCache.h"

namespace {

std::string tagStr(std::string_view taskTag) {
    return std::string{taskTag};
}

} // namespace

namespace yolo26 {

Yolo26TrtSession::Yolo26TrtSession() = default;

Yolo26TrtSession::~Yolo26TrtSession() {
    freeBuffers();
    releaseEngine();
}

void Yolo26TrtSession::releaseEngine() {
    if (context_ != nullptr) {
        delete context_;
        context_ = nullptr;
    }
    if (engine_ != nullptr) {
        delete engine_;
        engine_ = nullptr;
    }
}

void Yolo26TrtSession::freeBuffers() {
    cudaSetDevice(deviceId_);
    for (void*& p : inputDev_) {
        if (p != nullptr) {
            cudaFree(p);
            p = nullptr;
        }
    }
    inputDev_.clear();
    for (void*& p : outputDev_) {
        if (p != nullptr) {
            cudaFree(p);
            p = nullptr;
        }
    }
    outputDev_.clear();
    for (void*& p : outputHostPinned_) {
        if (p != nullptr) {
            cudaFreeHost(p);
            p = nullptr;
        }
    }
    outputHostPinned_.clear();
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

bool Yolo26TrtSession::allocateBuffers() {
    freeBuffers();
    if (inputsNum_ == 0 || outputsNum_ == 0) {
        return false;
    }
    cudaSetDevice(deviceId_);
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        std::cerr << "Yolo26TrtSession: cudaStreamCreate failed\n";
        return false;
    }
    inputDev_.resize(inputsNum_);
    outputDev_.resize(outputsNum_);
    outputHostPinned_.resize(outputsNum_);
    for (size_t i = 0; i < inputsNum_; i++) {
        if (cudaMalloc(&inputDev_[i], inputSizes_[i]) != cudaSuccess) {
            std::cerr << "Yolo26TrtSession: cudaMalloc input failed\n";
            freeBuffers();
            return false;
        }
    }
    for (size_t i = 0; i < outputsNum_; i++) {
        if (cudaMalloc(&outputDev_[i], outputSizes_[i]) != cudaSuccess) {
            std::cerr << "Yolo26TrtSession: cudaMalloc output failed\n";
            freeBuffers();
            return false;
        }
        if (cudaHostAlloc(&outputHostPinned_[i], outputSizes_[i], cudaHostAllocDefault) != cudaSuccess) {
            std::cerr << "Yolo26TrtSession: cudaHostAlloc output failed\n";
            freeBuffers();
            return false;
        }
    }
    return true;
}

void Yolo26TrtSession::retrieveNetInfo() {
    vecInputDims_.clear();
    vecInputLayerNames_.clear();
    inputSizes_.clear();
    vecOutputDims_.clear();
    vecOutputLayerNames_.clear();
    outputSizes_.clear();

    const int ioNumbers = engine_->getNbIOTensors();
    std::cout << "Yolo26TrtSession: number of io tensors: " << ioNumbers << std::endl;

    for (int i = 0; i < ioNumbers; i++) {
        const char* layerName = engine_->getIOTensorName(i);
        const nvinfer1::TensorIOMode type = engine_->getTensorIOMode(layerName);
        const nvinfer1::Dims dim = engine_->getTensorShape(layerName);

        if (type == nvinfer1::TensorIOMode::kINPUT) {
            vecInputDims_.push_back(dim);
            vecInputLayerNames_.emplace_back(layerName);
            std::cout << "  input: " << layerName << std::endl;
            size_t bufferSize = sizeof(float);
            for (int j = 0; j < dim.nbDims; j++) {
                std::cout << "\t dim" << j << " size: " << dim.d[j] << std::endl;
                bufferSize *= static_cast<size_t>(dim.d[j]);
            }
            inputSizes_.push_back(bufferSize);
        } else if (type == nvinfer1::TensorIOMode::kOUTPUT) {
            vecOutputDims_.push_back(dim);
            vecOutputLayerNames_.emplace_back(layerName);
            std::cout << "  output: " << layerName << std::endl;
            size_t bufferSize = sizeof(float);
            for (int j = 0; j < dim.nbDims; j++) {
                std::cout << "\t dim" << j << " size: " << dim.d[j] << std::endl;
                bufferSize *= static_cast<size_t>(dim.d[j]);
            }
            outputSizes_.push_back(bufferSize);
        }
    }
    inputsNum_ = vecInputDims_.size();
    outputsNum_ = vecOutputDims_.size();
}

bool Yolo26TrtSession::loadFromOnnx(const std::string& onnxPath, std::string_view taskTag) {
    freeBuffers();
    releaseEngine();

    const std::string t = tagStr(taskTag);
    const std::string enginePath = trt_engine_cache::enginePathFromOnnx(onnxPath);

    if (!trt_engine_cache::shouldRebuildEngine(onnxPath, enginePath)) {
        std::cout << "Loading TensorRT engine cache (" << t << "): " << enginePath << std::endl;
        if (trt_engine_cache::deserializeEngine(enginePath, logger_, engine_, context_)) {
            retrieveNetInfo();
            if (inputsNum_ != inputSizes_.size() || outputsNum_ != outputSizes_.size()) {
                std::cerr << "Yolo26TrtSession: io count mismatch (" << t << ")\n";
                releaseEngine();
                return false;
            }
            modelH_ = vecInputDims_[0].d[2];
            modelW_ = vecInputDims_[0].d[3];
            if (!allocateBuffers()) {
                std::cerr << "Yolo26TrtSession: allocateBuffers failed (" << t << ")\n";
                releaseEngine();
                return false;
            }
            return true;
        }
        std::cerr << "Engine cache load failed (" << t << "), rebuilding from ONNX...\n";
        releaseEngine();
    }

    std::ifstream onnxFilestream(onnxPath, std::ios::binary);
    if (!onnxFilestream.is_open()) {
        std::cerr << "Open onnx file failed: " << onnxPath << std::endl;
        return false;
    }

    onnxFilestream.seekg(0, std::ios::end);
    const size_t onnxSize = static_cast<size_t>(onnxFilestream.tellg());
    onnxFilestream.seekg(0, std::ios::beg);

    std::vector<char> onnxData(onnxSize);
    onnxFilestream.read(onnxData.data(), static_cast<std::streamsize>(onnxSize));
    onnxFilestream.close();

    nvinfer1::IBuilder* iBuilder = nvinfer1::createInferBuilder(logger_);
    const nvinfer1::NetworkDefinitionCreationFlags flags{
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};
    nvinfer1::INetworkDefinition* network = iBuilder->createNetworkV2(flags);
    nvinfer1::IBuilderConfig* builderConfig = iBuilder->createBuilderConfig();
    nvonnxparser::IParser* onnxParser = nvonnxparser::createParser(*network, logger_);
    if (!onnxParser->parse(onnxData.data(), onnxSize)) {
        std::cerr << "Parse onnx buffer failed (" << t << ")\n";
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        return false;
    }

    std::cout << "Building TensorRT engine (" << t << "), may take a long time...\n";
    engine_ = iBuilder->buildEngineWithConfig(*network, *builderConfig);
    if (engine_ == nullptr) {
        std::cerr << "TRT engine create failed (" << t << ")\n";
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        return false;
    }
    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) {
        std::cerr << "TRT context create failed (" << t << ")\n";
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        releaseEngine();
        return false;
    }
    std::cout << "Building environment finished (" << t << ")\n";

    if (trt_engine_cache::serializeEngineToFile(engine_, enginePath)) {
        std::cout << "Saved TensorRT engine cache (" << t << "): " << enginePath << std::endl;
    } else {
        std::cerr << "Warning: could not save .engine cache (" << t << ")\n";
    }

    retrieveNetInfo();

    delete onnxParser;
    delete network;
    delete builderConfig;
    delete iBuilder;

    if (inputsNum_ != inputSizes_.size() || outputsNum_ != outputSizes_.size()) {
        std::cerr << "Yolo26TrtSession: io count mismatch after build (" << t << ")\n";
        releaseEngine();
        return false;
    }

    modelH_ = vecInputDims_[0].d[2];
    modelW_ = vecInputDims_[0].d[3];
    if (!allocateBuffers()) {
        std::cerr << "Yolo26TrtSession: allocateBuffers failed after build (" << t << ")\n";
        releaseEngine();
        return false;
    }
    return true;
}

float* Yolo26TrtSession::outputHost(size_t i) {
    if (i >= outputHostPinned_.size()) {
        return nullptr;
    }
    return static_cast<float*>(outputHostPinned_[i]);
}

const float* Yolo26TrtSession::outputHost(size_t i) const {
    if (i >= outputHostPinned_.size()) {
        return nullptr;
    }
    return static_cast<const float*>(outputHostPinned_[i]);
}

bool Yolo26TrtSession::enqueue(const cv::Mat& nchwBlob) {
    if (!ready() || inputsNum_ == 0) {
        std::cerr << "Yolo26TrtSession::enqueue: not ready\n";
        return false;
    }
    if (nchwBlob.empty() || nchwBlob.type() != CV_32F) {
        std::cerr << "Yolo26TrtSession::enqueue: expect CV_32F NCHW blob\n";
        return false;
    }
    const size_t needBytes = inputSizes_[0];
    const size_t gotBytes = static_cast<size_t>(nchwBlob.total()) * nchwBlob.elemSize();
    if (needBytes != gotBytes) {
        std::cerr << "Yolo26TrtSession::enqueue: blob bytes mismatch (need " << needBytes << ", got " << gotBytes
                  << ")\n";
        return false;
    }

    cudaSetDevice(deviceId_);
    cudaMemcpyAsync(inputDev_[0], nchwBlob.data, inputSizes_[0], cudaMemcpyHostToDevice, stream_);
    for (size_t i = 0; i < inputsNum_; i++) {
        context_->setInputTensorAddress(vecInputLayerNames_[i].c_str(), inputDev_[i]);
    }
    for (size_t i = 0; i < outputsNum_; i++) {
        context_->setOutputTensorAddress(vecOutputLayerNames_[i].c_str(), outputDev_[i]);
    }

#if NV_TENSORRT_MAJOR >= 10
    context_->enqueueV3(stream_);
#else
#error "Yolo26TrtSession requires TensorRT 10+ (enqueueV3)."
#endif

    for (size_t i = 0; i < outputsNum_; i++) {
        cudaMemcpyAsync(outputHostPinned_[i], outputDev_[i], outputSizes_[i], cudaMemcpyDeviceToHost, stream_);
    }
    cudaStreamSynchronize(stream_);
    return true;
}

} // namespace yolo26
