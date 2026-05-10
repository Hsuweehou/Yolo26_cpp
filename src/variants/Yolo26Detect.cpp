#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "yolo26/Yolo26BackendFactory.h"
#include "yolo26/Yolo26Detect.h"
#include "yolo26/Yolo26Letterbox.h"

namespace fs = std::filesystem;

namespace {

inline float sigmoid1(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

} // namespace

Yolo26Detect::Yolo26Detect(const DetectConfig& config) {
    modelFile_ = config.modelFile;
    scoreThreshold_ = config.scoreThreshold;
    maxDetections_ = config.maxDetections;
    numClasses_ = config.numClasses;
    backendKind_ = config.backendKind;
}

Yolo26Detect::~Yolo26Detect() = default;

bool Yolo26Detect::init() {
    if (!fs::exists(fs::absolute(modelFile_))) {
        std::cerr << "Cannot find model file: " << modelFile_ << std::endl;
        return false;
    }

    backend_ = yolo26::CreateYolo26VariantBackend(backendKind_);
    if (!backend_) {
        std::cerr << "Yolo26Detect: failed to create backend (kind not supported?)\n";
        return false;
    }

    std::cout << "Try loading onnx file (detect): " << modelFile_ << std::endl;
    const bool ok = backend_->loadFromOnnx(fs::absolute(modelFile_).string(), "detect");
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
    letterboxUltralytics(image, backend_->modelWidth(), backend_->modelHeight(), letterboxed, letterboxScale_,
                         letterboxPadW_, letterboxPadH_);

    cv::Mat blob =
        cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0, cv::Size(backend_->modelWidth(), backend_->modelHeight()),
                                 cv::Scalar(), true, false, CV_32F);

    if (!backend_->enqueue(blob)) {
        return {};
    }

    return postProcessing(backend_->outputHost(0));
}

std::vector<DetectInferResult> Yolo26Detect::postProcessing(float* output0) const {
    if (!backend_ || backend_->numOutputs() != 1 || output0 == nullptr) {
        std::cerr << "Yolo26Detect: expect exactly 1 output tensor. outputs="
                  << (backend_ ? backend_->numOutputs() : 0) << std::endl;
        return {};
    }

    const nvinfer1::Dims& dim = backend_->outputDims()[0];
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
