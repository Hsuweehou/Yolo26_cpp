#ifndef YOLO26_YOLO26LETTERBOX_H
#define YOLO26_YOLO26LETTERBOX_H

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

inline void letterboxUltralytics(const cv::Mat& src, int dstW, int dstH, cv::Mat& dst, float& scale, float& padW,
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

#endif
