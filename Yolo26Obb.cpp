#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <NvOnnxParser.h>

#include "TrtEngineCache.h"
#include "Yolo26Obb.h"

namespace fs = std::filesystem;

namespace {

inline float sigmoid1(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

constexpr float kPi = 3.14159265f;

// ultralytics 同款 letterbox：等比、居中、灰边 114
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

Yolo26Obb::Yolo26Obb(const ObbConfig& config) {
    modelFile_ = config.modelFile;
    scoreThreshold_ = config.scoreThreshold;
    maxDetections_ = config.maxDetections;
    numClasses_ = config.numClasses;
    end2endLayout_ = config.end2endLayout;
    angleInRadians_ = config.angleInRadians;
    logger_ = new Yolo26ObbLogger();
}

Yolo26Obb::~Yolo26Obb() {
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

bool Yolo26Obb::init() {
    if (!fs::exists(fs::absolute(modelFile_))) {
        std::cerr << "Cannot find model file: " << modelFile_ << std::endl;
        return false;
    }

    std::cout << "Try loading onnx file (obb): " << modelFile_ << std::endl;
    const bool ok = initFromOnnx(fs::absolute(modelFile_).string());
    if (ok) {
        std::cout << "Loading succeed (obb)..." << std::endl;
    } else {
        std::cerr << "Loading failed (obb)..." << std::endl;
    }
    return ok;
}

std::vector<ObbInferResult> Yolo26Obb::inference(const cv::Mat& image) {
    imageWidth_ = image.cols;
    imageHeight_ = image.rows;

    cv::Mat letterboxed;
    letterboxUltralytics(image, modelWidth_, modelHeight_, letterboxed, letterboxScale_, letterboxPadW_,
                         letterboxPadH_);

    cv::Mat blob = cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(modelWidth_, modelHeight_),
                                           cv::Scalar(), true, false, CV_32F);

    cudaSetDevice(deviceId_);
    cudaStream_t stream{};
    cudaStreamCreate(&stream);

    void** outputHostBuffers = new void*[outputsNum_];
    for (size_t i = 0; i < outputsNum_; i++) {
        cudaHostAlloc(&outputHostBuffers[i], outputSizes_[i], 0);
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
#error "Yolo26Obb requires TensorRT 10+ (enqueueV3)."
#endif

    for (size_t i = 0; i < outputsNum_; i++) {
        cudaMemcpyAsync(outputHostBuffers[i], outputDeviceBuffers[i], outputSizes_[i], cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::vector<ObbInferenceOutput> inferOutputs;
    inferOutputs.reserve(outputsNum_);
    for (size_t i = 0; i < outputsNum_; i++) {
        ObbInferenceOutput out;
        out.data = std::shared_ptr<float>(static_cast<float*>(outputHostBuffers[i]),
                                          [](float* p) { cudaFreeHost(p); });
        out.size = static_cast<uint32_t>(outputSizes_[i]);
        inferOutputs.push_back(std::move(out));
    }
    delete[] outputHostBuffers;

    for (size_t i = 0; i < inputsNum_; i++) {
        cudaFree(inputDeviceBuffers[i]);
    }
    for (size_t i = 0; i < outputsNum_; i++) {
        cudaFree(outputDeviceBuffers[i]);
    }
    delete[] inputDeviceBuffers;
    delete[] outputDeviceBuffers;

    return postProcessing(inferOutputs);
}

bool Yolo26Obb::initFromOnnx(const std::string& onnxPath) {
    const std::string enginePath = trt_engine_cache::enginePathFromOnnx(onnxPath);

    if (!trt_engine_cache::shouldRebuildEngine(onnxPath, enginePath)) {
        std::cout << "Loading TensorRT engine cache (obb): " << enginePath << std::endl;
        if (trt_engine_cache::deserializeEngine(enginePath, *logger_, engine_, context_)) {
            retrieveNetInfo();
            if (inputsNum_ != inputSizes_.size() || outputsNum_ != outputSizes_.size()) {
                std::cerr << "Error network's input/output number (obb)..." << std::endl;
                return false;
            }
            modelHeight_ = vecInputDims_[0].d[2];
            modelWidth_ = vecInputDims_[0].d[3];
            return true;
        }
        std::cerr << "Engine cache load failed (obb), rebuilding from ONNX..." << std::endl;
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
        std::cerr << "Parse onnx buffer failed (obb).." << std::endl;
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        return false;
    }

    std::cout << "Building inference environment (obb), may take very long....(first build or ONNX newer)"
              << std::endl;
    engine_ = iBuilder->buildEngineWithConfig(*network, *builderConfig);
    if (engine_ == nullptr) {
        std::cerr << "TRT engine create failed (obb).." << std::endl;
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        return false;
    }
    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) {
        std::cerr << "TRT context create failed (obb).." << std::endl;
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        return false;
    }
    std::cout << "Building environment finished (obb)" << std::endl;

    if (trt_engine_cache::serializeEngineToFile(engine_, enginePath)) {
        std::cout << "Saved TensorRT engine cache (obb): " << enginePath << std::endl;
    } else {
        std::cerr << "Warning: could not save .engine cache (obb)" << std::endl;
    }

    retrieveNetInfo();

    delete onnxParser;
    delete network;
    delete builderConfig;
    delete iBuilder;

    if (inputsNum_ != inputSizes_.size() || outputsNum_ != outputSizes_.size()) {
        std::cerr << "Error network's input/output number (obb)..." << std::endl;
        return false;
    }

    modelHeight_ = vecInputDims_[0].d[2];
    modelWidth_ = vecInputDims_[0].d[3];
    return true;
}

void Yolo26Obb::retrieveNetInfo() {
    vecInputDims_.clear();
    vecInputLayerNames_.clear();
    inputSizes_.clear();
    vecOutputDims_.clear();
    vecOutputLayerNames_.clear();
    outputSizes_.clear();

    const int ioNumbers = engine_->getNbIOTensors();
    std::cout << "number of io layers (obb): " << ioNumbers << std::endl;

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

std::vector<ObbInferResult> Yolo26Obb::postProcessing(std::vector<ObbInferenceOutput>& inferOutputs) const {
    // 1×N×7：cx,cy,w,h,类得分,类id,角（不是 xyxy）
    if (outputsNum_ != 1 || inferOutputs.size() != 1) {
        std::cerr << "Yolo26Obb: expect exactly 1 output tensor. outputsNum_=" << outputsNum_ << std::endl;
        return {};
    }

    const nvinfer1::Dims& dim = vecOutputDims_[0];
    int d1 = 0;
    int d2 = 0;
    if (dim.nbDims == 3) {
        d1 = dim.d[1];
        d2 = dim.d[2];
    } else if (dim.nbDims == 4 && dim.d[0] == 1 && dim.d[1] == 1) {
        d1 = dim.d[2];
        d2 = dim.d[3];
        std::cout << "Yolo26Obb: output rank 4 [1,1,N,C], using N=" << d1 << " C=" << d2 << std::endl;
    } else {
        std::cerr << "Yolo26Obb: expect rank 3 [1,N,C]/[1,C,N] or rank 4 [1,1,N,C], got nbDims=" << dim.nbDims
                  << std::endl;
        return {};
    }

    constexpr int kBoxParams = 5;

    auto inferNumClassesFromC = [&](int C) -> int { return C - kBoxParams; };

    bool layout_cn = false;
    bool layout_nc = false;
    int numPred = 0;
    size_t featureDim = 0;
    size_t numClassesDecode = numClasses_;
    bool e2eUltralytics = false;

    const int expectedC = static_cast<int>(kBoxParams + numClasses_);

    if (end2endLayout_ && (d2 == 7 || d1 == 7)) {
        e2eUltralytics = true;
        if (d2 == 7) {
            layout_nc = true;
            numPred = d1;
            featureDim = 7;
        } else {
            layout_cn = true;
            numPred = d2;
            featureDim = 7;
        }
        std::cout << "Yolo26Obb: Ultralytics end-to-end [cx,cy,w,h, max_class_prob, class_id, angle], C="
                  << featureDim << std::endl;
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
        if (nc2 >= 1 && d2 == kBoxParams + nc2) {
            layout_nc = true;
            numPred = d1;
            featureDim = static_cast<size_t>(d2);
            numClassesDecode = static_cast<size_t>(nc2);
            std::cout << "Yolo26Obb: legacy layout C=5+nc, numClasses=" << numClassesDecode << " from C=" << d2
                      << std::endl;
        } else {
            const int nc1 = inferNumClassesFromC(d1);
            if (nc1 >= 1 && d1 == kBoxParams + nc1) {
                layout_cn = true;
                numPred = d2;
                featureDim = static_cast<size_t>(d1);
                numClassesDecode = static_cast<size_t>(nc1);
                std::cout << "Yolo26Obb: legacy layout C=5+nc, numClasses=" << numClassesDecode << " from C=" << d1
                          << std::endl;
            } else {
                std::cerr << "Yolo26Obb: cannot match feature dim. Expect C=7 (e2e) or C = 5 + numClasses (= "
                          << expectedC << "). d1=" << d1 << " d2=" << d2 << std::endl;
                return {};
            }
        }
    }

    auto* base = static_cast<float*>(inferOutputs[0].data.get());

    auto getFeat = [&](int detIdx, size_t k) -> float {
        if (layout_cn) {
            return base[k * static_cast<size_t>(numPred) + static_cast<size_t>(detIdx)];
        }
        return base[static_cast<size_t>(detIdx) * featureDim + k];
    };

    struct Cand {
        cv::RotatedRect rrect{};
        float score = 0.f;
        size_t cls = 0;
        int rawIndex = 0;
    };

    std::vector<Cand> cands;

    for (int i = 0; i < numPred; i++) {
        float score = 0.f;
        size_t bestCls = 0;
        cv::RotatedRect rr;

        if (e2eUltralytics) {
            const float cx = getFeat(i, 0);
            const float cy = getFeat(i, 1);
            const float w = getFeat(i, 2);
            const float h = getFeat(i, 3);
            score = getFeat(i, 4);
            if (score > 1.0f || score < 0.0f) {
                score = sigmoid1(score);
            }
            const float clsRaw = getFeat(i, 5);
            const float ang = getFeat(i, 6);
            const long clsRounded = std::lround(static_cast<double>(clsRaw));
            if (clsRounded < 0) {
                continue;
            }
            bestCls = static_cast<size_t>(clsRounded);
            if (numClasses_ == 1) {
                bestCls = 0;
            } else if (numClasses_ > 0 && bestCls >= numClasses_) {
                if (clsRounded >= 1 && clsRounded <= static_cast<long>(numClasses_)) {
                    bestCls = static_cast<size_t>(clsRounded - 1);
                } else {
                    continue;
                }
            }
            if (score < scoreThreshold_) {
                continue;
            }

            float mcx = cx;
            float mcy = cy;
            float mw = w;
            float mh = h;
            const float legMax = std::max(std::max(std::abs(cx), std::abs(cy)), std::max(std::abs(w), std::abs(h)));
            if (legMax <= 1.5f && std::abs(w) > 1e-6f && std::abs(h) > 1e-6f) {
                mcx *= static_cast<float>(modelWidth_);
                mcy *= static_cast<float>(modelHeight_);
                mw *= static_cast<float>(modelWidth_);
                mh *= static_cast<float>(modelHeight_);
            }
            const float cx_i = (mcx - letterboxPadW_) / letterboxScale_;
            const float cy_i = (mcy - letterboxPadH_) / letterboxScale_;
            const float w_i = std::abs(mw / letterboxScale_);
            const float h_i = std::abs(mh / letterboxScale_);
            if (w_i < 1.f || h_i < 1.f) {
                continue;
            }
            const float angleDeg = angleInRadians_ ? ang * (180.f / kPi) : ang;
            rr = cv::RotatedRect(cv::Point2f(cx_i, cy_i), cv::Size2f(w_i, h_i), angleDeg);
        } else {
            const float cx = getFeat(i, 0);
            const float cy = getFeat(i, 1);
            const float w = getFeat(i, 2);
            const float h = getFeat(i, 3);
            const float ang = getFeat(i, 4);

            if (numClassesDecode == 1) {
                score = getFeat(i, 5);
                if (score > 1.0f || score < 0.0f) {
                    score = sigmoid1(score);
                }
            } else {
                float best = -1e9f;
                for (size_t c = 0; c < numClassesDecode; c++) {
                    float v = getFeat(i, 5 + c);
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

            const float angleDeg = angleInRadians_ ? ang * (180.f / kPi) : ang;
            float mcx = cx;
            float mcy = cy;
            float mw = w;
            float mh = h;
            const float legMax = std::max(std::max(std::abs(cx), std::abs(cy)), std::max(std::abs(w), std::abs(h)));
            if (legMax <= 1.5f && std::abs(w) > 1e-6f && std::abs(h) > 1e-6f) {
                mcx *= static_cast<float>(modelWidth_);
                mcy *= static_cast<float>(modelHeight_);
                mw *= static_cast<float>(modelWidth_);
                mh *= static_cast<float>(modelHeight_);
            }
            const float cx_i = (mcx - letterboxPadW_) / letterboxScale_;
            const float cy_i = (mcy - letterboxPadH_) / letterboxScale_;
            const float w_i = std::abs(mw / letterboxScale_);
            const float h_i = std::abs(mh / letterboxScale_);

            if (w_i < 1.f || h_i < 1.f) {
                continue;
            }

            rr = cv::RotatedRect(cv::Point2f(cx_i, cy_i), cv::Size2f(w_i, h_i), angleDeg);
        }

        Cand cd;
        cd.rrect = rr;
        cd.score = score;
        cd.cls = bestCls;
        cd.rawIndex = i;
        cands.push_back(std::move(cd));
    }

    std::vector<ObbInferResult> out;
    out.reserve(cands.size());
    for (const Cand& c : cands) {
        ObbInferResult dr;
        dr.rrect = c.rrect;
        dr.score = c.score;
        dr.classIndex = c.cls;
        dr.index = static_cast<size_t>(c.rawIndex);
        out.push_back(std::move(dr));
    }

    std::sort(out.begin(), out.end(), [](const ObbInferResult& a, const ObbInferResult& b) {
        return a.score > b.score;
    });
    if (maxDetections_ > 0 && out.size() > maxDetections_) {
        out.resize(maxDetections_);
    }
    return out;
}
