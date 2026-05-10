#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "yolo26/Yolo26BackendFactory.h"
#include "yolo26/Yolo26Letterbox.h"
#include "yolo26/Yolo26Obb.h"

namespace fs = std::filesystem;

namespace {

inline float sigmoid1(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

constexpr float kPi = 3.14159265f;

} // namespace

Yolo26Obb::Yolo26Obb(const ObbConfig& config) {
    modelFile_ = config.modelFile;
    scoreThreshold_ = config.scoreThreshold;
    maxDetections_ = config.maxDetections;
    numClasses_ = config.numClasses;
    end2endLayout_ = config.end2endLayout;
    angleInRadians_ = config.angleInRadians;
    backendKind_ = config.backendKind;
}

Yolo26Obb::~Yolo26Obb() = default;

bool Yolo26Obb::init() {
    if (!fs::exists(fs::absolute(modelFile_))) {
        std::cerr << "Cannot find model file: " << modelFile_ << std::endl;
        return false;
    }

    backend_ = yolo26::CreateYolo26VariantBackend(backendKind_);
    if (!backend_) {
        std::cerr << "Yolo26Obb: failed to create backend\n";
        return false;
    }

    std::cout << "Try loading onnx file (obb): " << modelFile_ << std::endl;
    const bool ok = backend_->loadFromOnnx(fs::absolute(modelFile_).string(), "obb");
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

std::vector<ObbInferResult> Yolo26Obb::postProcessing(float* output0) const {
    if (!backend_ || backend_->numOutputs() != 1 || output0 == nullptr) {
        std::cerr << "Yolo26Obb: expect exactly 1 output tensor. outputs="
                  << (backend_ ? backend_->numOutputs() : 0) << std::endl;
        return {};
    }

    const nvinfer1::Dims& dim = backend_->outputDims()[0];
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

    auto* base = output0;

    auto getFeat = [&](int detIdx, size_t k) -> float {
        if (layout_cn) {
            return base[k * static_cast<size_t>(numPred) + static_cast<size_t>(detIdx)];
        }
        return base[static_cast<size_t>(detIdx) * featureDim + k];
    };

    const int mw = backend_->modelWidth();
    const int mh = backend_->modelHeight();

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
            float mw_box = w;
            float mh_box = h;
            const float legMax = std::max(std::max(std::abs(cx), std::abs(cy)), std::max(std::abs(w), std::abs(h)));
            if (legMax <= 1.5f && std::abs(w) > 1e-6f && std::abs(h) > 1e-6f) {
                mcx *= static_cast<float>(mw);
                mcy *= static_cast<float>(mh);
                mw_box *= static_cast<float>(mw);
                mh_box *= static_cast<float>(mh);
            }
            const float cx_i = (mcx - letterboxPadW_) / letterboxScale_;
            const float cy_i = (mcy - letterboxPadH_) / letterboxScale_;
            const float w_i = std::abs(mw_box / letterboxScale_);
            const float h_i = std::abs(mh_box / letterboxScale_);
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
            float mw_box = w;
            float mh_box = h;
            const float legMax = std::max(std::max(std::abs(cx), std::abs(cy)), std::max(std::abs(w), std::abs(h)));
            if (legMax <= 1.5f && std::abs(w) > 1e-6f && std::abs(h) > 1e-6f) {
                mcx *= static_cast<float>(mw);
                mcy *= static_cast<float>(mh);
                mw_box *= static_cast<float>(mw);
                mh_box *= static_cast<float>(mh);
            }
            const float cx_i = (mcx - letterboxPadW_) / letterboxScale_;
            const float cy_i = (mcy - letterboxPadH_) / letterboxScale_;
            const float w_i = std::abs(mw_box / letterboxScale_);
            const float h_i = std::abs(mh_box / letterboxScale_);

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
