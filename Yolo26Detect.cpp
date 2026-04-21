#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <NvOnnxParser.h>

#include "TrtEngineCache.h"
#include "Yolo26Detect.h"

namespace fs = std::filesystem;

namespace {

inline float sigmoid1(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ultralytics 同款 letterbox：等比缩放、居中、灰边 114
void letterboxUltralytics(const cv::Mat& src, int dstW, int dstH, cv::Mat& dst, float& scale, float& padW,
                          float& padH) {
    scale = std::min(static_cast<float>(dstW) / static_cast<float>(src.cols),
                     static_cast<float>(dstH) / static_cast<float>(src.rows));
    const int nw = static_cast<int>(std::round(static_cast<float>(src.cols) * scale));
    const int nh = static_cast<int>(std::round(static_cast<float>(src.rows) * scale));
    padW = (static_cast<float>(dstW) - static_cast<float>(nw)) * 0.5f;
    padH = (static_cast<float>(dstH) - static_cast<float>(nh)) * 0.5f;
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    dst = cv::Mat(dstH, dstW, src.type(), cv::Scalar(114, 114, 114));
    const int left = static_cast<int>(std::floor(padW));
    const int top = static_cast<int>(std::floor(padH));
    resized.copyTo(dst(cv::Rect(left, top, nw, nh)));
}

} // namespace

Yolo26Detect::Yolo26Detect(const DetectConfig& config) {
    modelFile_ = config.modelFile;
    scoreThreshold_ = config.scoreThreshold;
    maxDetections_ = config.maxDetections;
    numClasses_ = config.numClasses;
    logger_ = new Yolo26DetectLogger();
}

Yolo26Detect::~Yolo26Detect() {
    freeBuffers();
    if (context_ != nullptr) {
        delete context_;
        context_ = nullptr;
    }
    if (engine_ != nullptr) {
        delete engine_;
        engine_ = nullptr;
    }
    delete logger_;
    logger_ = nullptr;
}

bool Yolo26Detect::init() {
    if (!fs::exists(fs::absolute(modelFile_))) {
        std::cerr << "Cannot find model file: " << modelFile_ << std::endl;
        return false;
    }

    std::cout << "Try loading onnx file (detect): " << modelFile_ << std::endl;
    const bool ok = initFromOnnx(fs::absolute(modelFile_).string());
    if (ok) {
        std::cout << "Loading succeed (detect)..." << std::endl;
    } else {
        std::cerr << "Loading failed (detect)..." << std::endl;
    }
    return ok;
}

std::vector<DetectInferResult> Yolo26Detect::inference(const cv::Mat& image) {
    imageWidth_ = image.cols;
    imageHeight_ = image.rows;

    cv::Mat letterboxed;
    letterboxUltralytics(image, modelWidth_, modelHeight_, letterboxed, letterboxScale_, letterboxPadW_,
                         letterboxPadH_);

    cv::Mat blob = cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(modelWidth_, modelHeight_),
                                           cv::Scalar(), true, false, CV_32F);

    cudaSetDevice(deviceId_);
    if (stream_ == nullptr || inputDev_.size() != inputsNum_ || outputDev_.size() != outputsNum_) {
        std::cerr << "Yolo26Detect: inference buffers not allocated" << std::endl;
        return {};
    }

    cudaMemcpyAsync(inputDev_[0], blob.data, inputSizes_[0], cudaMemcpyHostToDevice, stream_);
    for (size_t i = 0; i < inputsNum_; i++) {
        context_->setInputTensorAddress(vecInputLayerNames_[i].c_str(), inputDev_[i]);
    }
    for (size_t i = 0; i < outputsNum_; i++) {
        context_->setOutputTensorAddress(vecOutputLayerNames_[i].c_str(), outputDev_[i]);
    }

#if NV_TENSORRT_MAJOR >= 10
    context_->enqueueV3(stream_);
#else
#error "Yolo26Detect requires TensorRT 10+ (enqueueV3)."
#endif

    for (size_t i = 0; i < outputsNum_; i++) {
        cudaMemcpyAsync(outputHostPinned_[i], outputDev_[i], outputSizes_[i], cudaMemcpyDeviceToHost, stream_);
    }
    cudaStreamSynchronize(stream_);

    return postProcessing(static_cast<float*>(outputHostPinned_[0]));
}

bool Yolo26Detect::initFromOnnx(const std::string& onnxPath) {
    const std::string enginePath = trt_engine_cache::enginePathFromOnnx(onnxPath);

    if (!trt_engine_cache::shouldRebuildEngine(onnxPath, enginePath)) {
        std::cout << "Loading TensorRT engine cache (detect): " << enginePath << std::endl;
        if (trt_engine_cache::deserializeEngine(enginePath, *logger_, engine_, context_)) {
            retrieveNetInfo();
            if (inputsNum_ != inputSizes_.size() || outputsNum_ != outputSizes_.size()) {
                std::cerr << "Error network's input/output number (detect)..." << std::endl;
                return false;
            }
            modelHeight_ = vecInputDims_[0].d[2];
            modelWidth_ = vecInputDims_[0].d[3];
            if (!allocateBuffers()) {
                std::cerr << "allocateBuffers failed (detect)" << std::endl;
                return false;
            }
            return true;
        }
        std::cerr << "Engine cache load failed (detect), rebuilding from ONNX..." << std::endl;
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
        std::cerr << "Open onnx file failed: " << onnxPath << std::endl;
        return false;
    }

    onnxFilestream.seekg(0, std::ios::end);
    const size_t onnxSize = static_cast<size_t>(onnxFilestream.tellg());
    onnxFilestream.seekg(0, std::ios::beg);

    std::vector<char> onnxData(onnxSize);
    onnxFilestream.read(onnxData.data(), static_cast<std::streamsize>(onnxSize));
    onnxFilestream.close();

    nvinfer1::IBuilder* iBuilder = nvinfer1::createInferBuilder(*logger_);
    const nvinfer1::NetworkDefinitionCreationFlags flags{
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};
    nvinfer1::INetworkDefinition* network = iBuilder->createNetworkV2(flags);
    nvinfer1::IBuilderConfig* builderConfig = iBuilder->createBuilderConfig();
    nvonnxparser::IParser* onnxParser = nvonnxparser::createParser(*network, *logger_);
    if (!onnxParser->parse(onnxData.data(), onnxSize)) {
        std::cerr << "Parse onnx buffer failed (detect).." << std::endl;
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        return false;
    }

    std::cout << "Building inference environment (detect), may take very long....(first build or ONNX newer)"
              << std::endl;
    engine_ = iBuilder->buildEngineWithConfig(*network, *builderConfig);
    if (engine_ == nullptr) {
        std::cerr << "TRT engine create failed (detect).." << std::endl;
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        return false;
    }
    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) {
        std::cerr << "TRT context create failed (detect).." << std::endl;
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        return false;
    }
    std::cout << "Building environment finished (detect)" << std::endl;

    if (trt_engine_cache::serializeEngineToFile(engine_, enginePath)) {
        std::cout << "Saved TensorRT engine cache (detect): " << enginePath << std::endl;
    } else {
        std::cerr << "Warning: could not save .engine cache (detect)" << std::endl;
    }

    retrieveNetInfo();

    delete onnxParser;
    delete network;
    delete builderConfig;
    delete iBuilder;

    if (inputsNum_ != inputSizes_.size() || outputsNum_ != outputSizes_.size()) {
        std::cerr << "Error network's input/output number (detect)..." << std::endl;
        return false;
    }

    modelHeight_ = vecInputDims_[0].d[2];
    modelWidth_ = vecInputDims_[0].d[3];
    if (!allocateBuffers()) {
        std::cerr << "allocateBuffers failed (detect)" << std::endl;
        return false;
    }
    return true;
}

void Yolo26Detect::freeBuffers() {
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

bool Yolo26Detect::allocateBuffers() {
    freeBuffers();
    if (inputsNum_ == 0 || outputsNum_ == 0) {
        return false;
    }
    cudaSetDevice(deviceId_);
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        std::cerr << "Yolo26Detect: cudaStreamCreate failed" << std::endl;
        return false;
    }
    inputDev_.resize(inputsNum_);
    outputDev_.resize(outputsNum_);
    outputHostPinned_.resize(outputsNum_);
    for (size_t i = 0; i < inputsNum_; i++) {
        if (cudaMalloc(&inputDev_[i], inputSizes_[i]) != cudaSuccess) {
            std::cerr << "Yolo26Detect: cudaMalloc input failed" << std::endl;
            freeBuffers();
            return false;
        }
    }
    for (size_t i = 0; i < outputsNum_; i++) {
        if (cudaMalloc(&outputDev_[i], outputSizes_[i]) != cudaSuccess) {
            std::cerr << "Yolo26Detect: cudaMalloc output failed" << std::endl;
            freeBuffers();
            return false;
        }
        if (cudaHostAlloc(&outputHostPinned_[i], outputSizes_[i], cudaHostAllocDefault) != cudaSuccess) {
            std::cerr << "Yolo26Detect: cudaHostAlloc output failed" << std::endl;
            freeBuffers();
            return false;
        }
    }
    return true;
}

void Yolo26Detect::retrieveNetInfo() {
    vecInputDims_.clear();
    vecInputLayerNames_.clear();
    inputSizes_.clear();
    vecOutputDims_.clear();
    vecOutputLayerNames_.clear();
    outputSizes_.clear();

    const int ioNumbers = engine_->getNbIOTensors();
    std::cout << "number of io layers (detect): " << ioNumbers << std::endl;

    for (int i = 0; i < ioNumbers; i++) {
        const char* layerName = engine_->getIOTensorName(i);
        const nvinfer1::TensorIOMode type = engine_->getTensorIOMode(layerName);
        const nvinfer1::Dims dim = engine_->getTensorShape(layerName);

        if (type == nvinfer1::TensorIOMode::kINPUT) {
            vecInputDims_.push_back(dim);
            vecInputLayerNames_.emplace_back(layerName);
            std::cout << "input layer: " << layerName << std::endl;
            size_t bufferSize = sizeof(float);
            for (int j = 0; j < dim.nbDims; j++) {
                std::cout << "\t dim" << j << " size: " << dim.d[j] << std::endl;
                bufferSize *= static_cast<size_t>(dim.d[j]);
            }
            inputSizes_.push_back(bufferSize);
        } else if (type == nvinfer1::TensorIOMode::kOUTPUT) {
            vecOutputDims_.push_back(dim);
            vecOutputLayerNames_.emplace_back(layerName);
            std::cout << "output layer: " << layerName << std::endl;
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

std::vector<DetectInferResult> Yolo26Detect::postProcessing(float* output0) const {
    // 1×N×6：xyxy + conf + class，网络里已做 NMS
    if (outputsNum_ != 1 || output0 == nullptr) {
        std::cerr << "Yolo26Detect: expect exactly 1 output tensor. outputsNum_=" << outputsNum_ << std::endl;
        return {};
    }

    const nvinfer1::Dims& dim = vecOutputDims_[0];
    if (dim.nbDims != 3) {
        std::cerr << "Yolo26Detect: expect output rank 3 [1,N,C] or [1,C,N], got nbDims=" << dim.nbDims << std::endl;
        return {};
    }

    const int d1 = dim.d[1];
    const int d2 = dim.d[2];

    auto inferNumClassesFromC = [&](int C) -> int { return C - 4; };

    bool layout_cn = false;
    bool layout_nc = false;
    int numPred = 0;
    size_t featureDim = 0;
    size_t numClassesDecode = numClasses_;
    bool e2eUltralytics = false;

    const int expectedC = static_cast<int>(4 + numClasses_);

    if (d2 == 6 || d1 == 6) {
        e2eUltralytics = true;
        if (d2 == 6) {
            layout_nc = true;
            numPred = d1;
            featureDim = 6;
        } else {
            layout_cn = true;
            numPred = d2;
            featureDim = 6;
        }
        std::cout << "Yolo26Detect: Ultralytics end-to-end layout [x1,y1,x2,y2, conf, class_id], C=" << featureDim
                  << std::endl;
    } else if (d2 == expectedC) {
        layout_nc = true;
        numPred = d1;
        featureDim = static_cast<size_t>(d2);
    } else if (d1 == expectedC) {
        layout_cn = true;
        numPred = d2;
        featureDim = static_cast<size_t>(d1);
    } else {
        const int nc2 = inferNumClassesFromC(d2);
        if (nc2 >= 1 && d2 == 4 + nc2) {
            layout_nc = true;
            numPred = d1;
            featureDim = static_cast<size_t>(d2);
            numClassesDecode = static_cast<size_t>(nc2);
            std::cout << "Yolo26Detect: legacy layout C=4+nc, numClasses=" << numClassesDecode << " from C=" << d2
                      << std::endl;
        } else {
            const int nc1 = inferNumClassesFromC(d1);
            if (nc1 >= 1 && d1 == 4 + nc1) {
                layout_cn = true;
                numPred = d2;
                featureDim = static_cast<size_t>(d1);
                numClassesDecode = static_cast<size_t>(nc1);
                std::cout << "Yolo26Detect: legacy layout C=4+nc, numClasses=" << numClassesDecode << " from C=" << d1
                          << std::endl;
            } else {
                std::cerr << "Yolo26Detect: cannot match feature dim. Expect C=6 (e2e) or C = 4 + numClasses (= "
                          << expectedC << " with current config). d1=" << d1 << " d2=" << d2 << std::endl;
                return {};
            }
        }
    }

    auto* base = output0;

    auto getFeat = [&](int detIdx, size_t k) -> float {
        if (layout_cn) {
            return base[k * static_cast<size_t>(numPred) + static_cast<size_t>(detIdx)];
        }
        return base[static_cast<size_t>(detIdx) * featureDim + k];
    };

    struct Cand {
        cv::Rect box{};
        float score = 0.f;
        size_t cls = 0;
        int rawIndex = 0;
    };

    std::vector<Cand> cands;

    for (int i = 0; i < numPred; i++) {
        const float x1 = getFeat(i, 0);
        const float y1 = getFeat(i, 1);
        const float x2 = getFeat(i, 2);
        const float y2 = getFeat(i, 3);

        float score = 0.f;
        size_t bestCls = 0;

        if (e2eUltralytics) {
            score = getFeat(i, 4);
            if (score > 1.0f || score < 0.0f) {
                score = sigmoid1(score);
            }
            const float clsRaw = getFeat(i, 5);
            const long clsRounded = std::lround(static_cast<double>(clsRaw));
            if (clsRounded < 0) {
                continue;
            }
            bestCls = static_cast<size_t>(clsRounded);
            if (numClasses_ > 0 && bestCls >= numClasses_) {
                continue;
            }
        } else if (numClassesDecode == 1) {
            score = getFeat(i, 4);
            if (score > 1.0f || score < 0.0f) {
                score = sigmoid1(score);
            }
        } else {
            float best = -1e9f;
            for (size_t c = 0; c < numClassesDecode; c++) {
                float v = getFeat(i, 4 + c);
                if (v > 1.0f || v < 0.0f) {
                    v = sigmoid1(v);
                }
                if (v > best) {
                    best = v;
                    bestCls = c;
                }
            }
            score = best;
        }

        if (score < scoreThreshold_) {
            continue;
        }

        const float sx = letterboxScale_;
        const float px = letterboxPadW_;
        const float py = letterboxPadH_;
        const float x1o = (x1 - px) / sx;
        const float y1o = (y1 - py) / sx;
        const float x2o = (x2 - px) / sx;
        const float y2o = (y2 - py) / sx;
        const int ix1 = static_cast<int>(std::floor(x1o));
        const int iy1 = static_cast<int>(std::floor(y1o));
        const int ix2 = static_cast<int>(std::ceil(x2o));
        const int iy2 = static_cast<int>(std::ceil(y2o));
        cv::Rect r(cv::Point(ix1, iy1), cv::Point(ix2, iy2));
        r &= cv::Rect(0, 0, imageWidth_, imageHeight_);
        if (r.width <= 0 || r.height <= 0) {
            continue;
        }

        Cand cd;
        cd.box = r;
        cd.score = score;
        cd.cls = bestCls;
        cd.rawIndex = i;
        cands.push_back(std::move(cd));
    }

    std::vector<DetectInferResult> out;
    out.reserve(cands.size());
    for (const Cand& c : cands) {
        DetectInferResult dr;
        dr.rect = c.box;
        dr.score = c.score;
        dr.classIndex = c.cls;
        dr.index = static_cast<size_t>(c.rawIndex);
        out.push_back(std::move(dr));
    }

    std::sort(out.begin(), out.end(), [](const DetectInferResult& a, const DetectInferResult& b) {
        return a.score > b.score;
    });
    if (maxDetections_ > 0 && out.size() > maxDetections_) {
        out.resize(maxDetections_);
    }
    return out;
}
