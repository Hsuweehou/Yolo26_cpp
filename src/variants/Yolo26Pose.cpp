#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "yolo26/Yolo26BackendFactory.h"
#include "yolo26/Yolo26Letterbox.h"
#include "yolo26/Yolo26Pose.h"

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
    backendKind_ = config.backendKind;
}

Yolo26Pose::~Yolo26Pose() = default;

bool Yolo26Pose::init() {
    if (!fs::exists(fs::absolute(modelFile_))) {
        std::cerr << "Cannot find model file: " << modelFile_ << std::endl;
        return false;
    }

    backend_ = yolo26::CreateYolo26VariantBackend(backendKind_);
    if (!backend_) {
        std::cerr << "Yolo26Pose: failed to create backend\n";
        return false;
    }

    std::cout << "Try loading onnx file (pose): " << modelFile_ << std::endl;
    const bool ok = backend_->loadFromOnnx(fs::absolute(modelFile_).string(), "pose");
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

std::vector<PoseInferResult> Yolo26Pose::postProcessing(float* output0) const {
    if (!backend_ || backend_->numOutputs() != 1 || output0 == nullptr) {
        std::cerr << "Yolo26Pose: expect exactly 1 output tensor (pose head). outputs="
                  << (backend_ ? backend_->numOutputs() : 0) << std::endl;
        return {};
    }

    const nvinfer1::Dims& dim = backend_->outputDims()[0];
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

    auto* base = output0;

    auto getFeat = [&](int detIdx, size_t k) -> float {
        if (layout_cn) {
            return base[k * static_cast<size_t>(numPred) + static_cast<size_t>(detIdx)];
        }
        return base[static_cast<size_t>(detIdx) * featureDim + k];
    };

    const float sx = letterboxScale_;
    const float px = letterboxPadW_;
    const float py = letterboxPadH_;

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
            pk.x = (kx - px) / sx;
            pk.y = (ky - py) / sx;
            pk.conf = kc;
            kpts.push_back(pk);
        }

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
