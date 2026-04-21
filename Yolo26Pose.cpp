#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <NvOnnxParser.h>

#include "TrtEngineCache.h"
#include "Yolo26Pose.h"

namespace fs = std::filesystem;

namespace {

inline float sigmoid1(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

} // namespace

Yolo26Pose::Yolo26Pose(const PoseConfig& config) {
    modelFile_ = config.modelFile;
    scoreThreshold_ = config.scoreThreshold;
    nmsThreshold_ = config.nmsThreshold;
    numKeypoints_ = config.numKeypoints;
    numClasses_ = config.numClasses;
    end2endLayout_ = config.end2endLayout;
    maxDetections_ = config.maxDetections;
    logger_ = new Yolo26PoseLogger();
}

Yolo26Pose::~Yolo26Pose() {
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

bool Yolo26Pose::init() {
    if (!fs::exists(fs::absolute(modelFile_))) {
        std::cerr << "Cannot find model file: " << modelFile_ << std::endl;
        return false;
    }

    std::cout << "Try loading onnx file (pose): " << modelFile_ << std::endl;
    const bool ok = initFromOnnx(fs::absolute(modelFile_).string());
    if (ok) {
        std::cout << "Loading succeed (pose)..." << std::endl;
    } else {
        std::cerr << "Loading failed (pose)..." << std::endl;
    }
    return ok;
}

std::vector<PoseInferResult> Yolo26Pose::inference(const cv::Mat& image) {
    imageWidth_ = image.cols;
    imageHeight_ = image.rows;

    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(modelWidth_, modelHeight_), cv::Scalar(), true,
                                          false, CV_32F);

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
#error "Yolo26Pose requires TensorRT 10+ (enqueueV3)."
#endif

    for (size_t i = 0; i < outputsNum_; i++) {
        cudaMemcpyAsync(outputHostBuffers[i], outputDeviceBuffers[i], outputSizes_[i], cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::vector<PoseInferenceOutput> inferOutputs;
    inferOutputs.reserve(outputsNum_);
    for (size_t i = 0; i < outputsNum_; i++) {
        PoseInferenceOutput out;
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

bool Yolo26Pose::initFromOnnx(const std::string& onnxPath) {
    const std::string enginePath = trt_engine_cache::enginePathFromOnnx(onnxPath);

    if (!trt_engine_cache::shouldRebuildEngine(onnxPath, enginePath)) {
        std::cout << "Loading TensorRT engine cache (pose): " << enginePath << std::endl;
        if (trt_engine_cache::deserializeEngine(enginePath, *logger_, engine_, context_)) {
            retrieveNetInfo();
            if (inputsNum_ != inputSizes_.size() || outputsNum_ != outputSizes_.size()) {
                std::cerr << "Error network's input/output number (pose)..." << std::endl;
                return false;
            }
            modelHeight_ = vecInputDims_[0].d[2];
            modelWidth_ = vecInputDims_[0].d[3];
            return true;
        }
        std::cerr << "Engine cache load failed (pose), rebuilding from ONNX..." << std::endl;
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
        std::cerr << "Parse onnx buffer failed (pose).." << std::endl;
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        return false;
    }

    std::cout << "Building inference environment (pose), may take very long....(first build or ONNX newer)"
              << std::endl;
    engine_ = iBuilder->buildEngineWithConfig(*network, *builderConfig);
    if (engine_ == nullptr) {
        std::cerr << "TRT engine create failed (pose).." << std::endl;
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        return false;
    }
    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) {
        std::cerr << "TRT context create failed (pose).." << std::endl;
        delete onnxParser;
        delete network;
        delete builderConfig;
        delete iBuilder;
        return false;
    }
    std::cout << "Building environment finished (pose)" << std::endl;

    if (trt_engine_cache::serializeEngineToFile(engine_, enginePath)) {
        std::cout << "Saved TensorRT engine cache (pose): " << enginePath << std::endl;
    } else {
        std::cerr << "Warning: could not save .engine cache (pose)" << std::endl;
    }

    retrieveNetInfo();

    delete onnxParser;
    delete network;
    delete builderConfig;
    delete iBuilder;

    if (inputsNum_ != inputSizes_.size() || outputsNum_ != outputSizes_.size()) {
        std::cerr << "Error network's input/output number (pose)..." << std::endl;
        return false;
    }

    modelHeight_ = vecInputDims_[0].d[2];
    modelWidth_ = vecInputDims_[0].d[3];
    return true;
}

void Yolo26Pose::retrieveNetInfo() {
    vecInputDims_.clear();
    vecInputLayerNames_.clear();
    inputSizes_.clear();
    vecOutputDims_.clear();
    vecOutputLayerNames_.clear();
    outputSizes_.clear();

    const int ioNumbers = engine_->getNbIOTensors();
    std::cout << "number of io layers (pose): " << ioNumbers << std::endl;

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

std::vector<PoseInferResult> Yolo26Pose::postProcessing(std::vector<PoseInferenceOutput>& inferOutputs) const {
    // end2end：C=6+K*3，和 detect 前 6 维同一套
    if (outputsNum_ != 1 || inferOutputs.size() != 1) {
        std::cerr << "Yolo26Pose: expect exactly 1 output tensor (pose head). outputsNum_=" << outputsNum_
                  << std::endl;
        return {};
    }

    const nvinfer1::Dims& dim = vecOutputDims_[0];
    if (dim.nbDims != 3) {
        std::cerr << "Yolo26Pose: expect output rank 3 [1,N,C] or [1,C,N], got nbDims=" << dim.nbDims << std::endl;
        return {};
    }

    const int d1 = dim.d[1];
    const int d2 = dim.d[2];
    const int kptBlock = static_cast<int>(numKeypoints_ * 3);
    const int e2eC = 6 + kptBlock;

    auto inferNumClassesFromC = [&](int C) -> int { return C - 4 - kptBlock; };

    bool layout_cn = false;
    bool layout_nc = false;
    int numPred = 0;
    size_t featureDim = 0;
    size_t numClassesDecode = numClasses_;
    bool e2eUltralytics = false;

    const int expectedLegacyC = static_cast<int>(4 + numClasses_ + kptBlock);

    if (end2endLayout_ && (d2 == e2eC || d1 == e2eC)) {
        e2eUltralytics = true;
        if (d2 == e2eC) {
            layout_nc = true;
            numPred = d1;
            featureDim = static_cast<size_t>(e2eC);
        } else {
            layout_cn = true;
            numPred = d2;
            featureDim = static_cast<size_t>(e2eC);
        }
        std::cout << "Yolo26Pose: end-to-end layout C=" << e2eC << " = [xyxy, conf, class_id] + " << numKeypoints_
                  << "*3 kpts (cfg/models/26/yolo26-pose.yaml)" << std::endl;
    } else if (d2 == expectedLegacyC) {
        layout_nc = true;
        numPred = d1;
        featureDim = static_cast<size_t>(d2);
    } else if (d1 == expectedLegacyC) {
        layout_cn = true;
        numPred = d2;
        featureDim = static_cast<size_t>(d1);
    } else {
        const int nc2 = inferNumClassesFromC(d2);
        if (nc2 >= 1 && d2 == 4 + nc2 + kptBlock) {
            layout_nc = true;
            numPred = d1;
            featureDim = static_cast<size_t>(d2);
            numClassesDecode = static_cast<size_t>(nc2);
            std::cout << "Yolo26Pose: legacy layout, auto-inferred numClasses=" << numClassesDecode << " from C=" << d2
                      << std::endl;
        } else {
            const int nc1 = inferNumClassesFromC(d1);
            if (nc1 >= 1 && d1 == 4 + nc1 + kptBlock) {
                layout_cn = true;
                numPred = d2;
                featureDim = static_cast<size_t>(d1);
                numClassesDecode = static_cast<size_t>(nc1);
                std::cout << "Yolo26Pose: legacy layout, auto-inferred numClasses=" << numClassesDecode << " from C=" << d1
                          << std::endl;
            } else {
                std::cerr << "Yolo26Pose: cannot match feature dim. Expect e2e C=" << e2eC << " or legacy C=4+nc+K*3 (= "
                          << expectedLegacyC << " with config). d1=" << d1 << " d2=" << d2 << std::endl;
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
        cv::Rect box;
        float score = 0.f;
        size_t cls = 0;
        std::vector<PoseKeypoint> kpts;
        int rawIndex = 0;
    };

    std::vector<cv::Rect> nmsBoxes;
    std::vector<float> nmsScores;
    std::vector<Cand> cands;

    for (int i = 0; i < numPred; i++) {
        const float x1 = getFeat(i, 0);
        const float y1 = getFeat(i, 1);
        const float x2 = getFeat(i, 2);
        const float y2 = getFeat(i, 3);

        float score = 0.f;
        size_t bestCls = 0;
        size_t kptBase = 0;

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
            kptBase = 6;
        } else if (numClassesDecode == 1) {
            score = getFeat(i, 4);
            if (score > 1.0f || score < 0.0f) {
                score = sigmoid1(score);
            }
            kptBase = 5;
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
            kptBase = 4 + numClassesDecode;
        }

        if (score < scoreThreshold_) {
            continue;
        }

        std::vector<PoseKeypoint> kpts;
        kpts.reserve(numKeypoints_);
        for (size_t k = 0; k < numKeypoints_; k++) {
            float kx = getFeat(i, kptBase + k * 3);
            float ky = getFeat(i, kptBase + k * 3 + 1);
            float kc = getFeat(i, kptBase + k * 3 + 2);
            if (kc > 1.0f || kc < 0.0f) {
                kc = sigmoid1(kc);
            }

            PoseKeypoint pk;
            pk.x = kx / static_cast<float>(modelWidth_) * static_cast<float>(imageWidth_);
            pk.y = ky / static_cast<float>(modelHeight_) * static_cast<float>(imageHeight_);
            pk.conf = kc;
            kpts.push_back(pk);
        }

        const int ix1 = static_cast<int>(x1 / static_cast<float>(modelWidth_) * static_cast<float>(imageWidth_));
        const int iy1 = static_cast<int>(y1 / static_cast<float>(modelHeight_) * static_cast<float>(imageHeight_));
        const int ix2 = static_cast<int>(x2 / static_cast<float>(modelWidth_) * static_cast<float>(imageWidth_));
        const int iy2 = static_cast<int>(y2 / static_cast<float>(modelHeight_) * static_cast<float>(imageHeight_));
        cv::Rect r(cv::Point(ix1, iy1), cv::Point(ix2, iy2));
        r &= cv::Rect(0, 0, imageWidth_, imageHeight_);

        Cand cd;
        cd.box = r;
        cd.score = score;
        cd.cls = bestCls;
        cd.kpts = std::move(kpts);
        cd.rawIndex = i;

        if (e2eUltralytics) {
            cands.push_back(std::move(cd));
        } else {
            nmsBoxes.push_back(r);
            nmsScores.push_back(score);
            cands.push_back(std::move(cd));
        }
    }

    if (e2eUltralytics) {
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
        if (maxDetections_ > 0 && cands.size() > maxDetections_) {
            cands.resize(maxDetections_);
        }
        std::vector<PoseInferResult> out;
        out.reserve(cands.size());
        for (const Cand& c : cands) {
            PoseInferResult r;
            r.rect = c.box;
            r.keypoints = c.kpts;
            r.score = c.score;
            r.classIndex = c.cls;
            r.index = static_cast<size_t>(c.rawIndex);
            out.push_back(std::move(r));
        }
        return out;
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(nmsBoxes, nmsScores, scoreThreshold_, nmsThreshold_, keep);

    std::vector<PoseInferResult> out;
    out.reserve(keep.size());
    for (int idx : keep) {
        if (idx < 0 || idx >= static_cast<int>(cands.size())) {
            continue;
        }
        const Cand& c = cands[static_cast<size_t>(idx)];
        PoseInferResult r;
        r.rect = c.box;
        r.keypoints = c.kpts;
        r.score = c.score;
        r.classIndex = c.cls;
        r.index = static_cast<size_t>(c.rawIndex);
        out.push_back(std::move(r));
    }
    return out;
}
