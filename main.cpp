#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "Yolo26Detect.h"
#include "Yolo26Obb.h"
#include "Yolo26Pose.h"
#include "Yolo26Seg.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static void initConsoleUtf8() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

namespace fs = std::filesystem;

// exe 所在目录；从 build/Release 运行时靠它找 cfg/datasets
static fs::path getExecutableDirectory() {
#if defined(_WIN32)
    std::wstring buf(MAX_PATH * 2, L'\0');
    DWORD len = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (len == 0) {
        return {};
    }
    while (len >= buf.size() - 1) {
        buf.resize(buf.size() * 2);
        len = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    }
    if (len == 0) {
        return {};
    }
    buf.resize(len);
    return fs::path(buf).parent_path();
#elif defined(__linux__)
    std::error_code ec;
    const fs::path self = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::path{} : self.parent_path();
#else
    return {};
#endif
}

// 把类别文件路径补成真实路径（在 build 里跑时常写错相对路径）
static std::string resolveClassNamesInputPath(const std::string& userPath) {
    std::error_code ec;
    const fs::path p(userPath);
    const fs::path fname = p.filename();

    auto tryCanon = [&](const fs::path& q) -> std::string {
        if (q.empty()) {
            return {};
        }
        if (!fs::exists(q, ec)) {
            return {};
        }
        return fs::weakly_canonical(q, ec).string();
    };

    if (std::string r = tryCanon(fs::absolute(p, ec)); !r.empty()) {
        return r;
    }
    if (std::string r = tryCanon(fs::current_path() / p); !r.empty()) {
        return r;
    }

    if (!fname.empty() && fname != "." && fname != "..") {
        if (std::string r = tryCanon(fs::current_path() / "cfg" / "datasets" / fname); !r.empty()) {
            return r;
        }
        const fs::path exeDir = getExecutableDirectory();
        if (!exeDir.empty()) {
            if (std::string r = tryCanon(exeDir / ".." / ".." / "cfg" / "datasets" / fname); !r.empty()) {
                return r;
            }
            if (std::string r = tryCanon(exeDir / ".." / ".." / ".." / "cfg" / "datasets" / fname); !r.empty()) {
                return r;
            }
        }
    }

    return userPath;
}

// 图像路径，规则跟类别文件差不多（cwd 不对时按 exe 来解析）
static std::string resolveImageInputPath(const std::string& userPath) {
    std::error_code ec;
    const fs::path p(userPath);
    const fs::path fname = p.filename();

    auto tryResolved = [&](const fs::path& q) -> std::string {
        if (q.empty()) {
            return {};
        }
        const fs::path canon = fs::weakly_canonical(q, ec);
        if (ec || !fs::exists(canon, ec) || fs::is_directory(canon, ec)) {
            return {};
        }
        return canon.string();
    };

    if (std::string r = tryResolved(fs::absolute(p, ec)); !r.empty()) {
        return r;
    }
    if (std::string r = tryResolved(fs::current_path() / p); !r.empty()) {
        return r;
    }

    const fs::path exeDir = getExecutableDirectory();
    if (!exeDir.empty()) {
        if (std::string r = tryResolved(exeDir / p); !r.empty()) {
            return r;
        }
        const fs::path projectRoot = exeDir.parent_path().parent_path();
        if (!fname.empty() && fname != "." && fname != "..") {
            if (std::string r = tryResolved(projectRoot / "cfg" / "datasets" / fname); !r.empty()) {
                return r;
            }
            if (std::string r = tryResolved(fs::current_path() / "cfg" / "datasets" / fname); !r.empty()) {
                return r;
            }
            if (std::string r = tryResolved(exeDir / ".." / ".." / "cfg" / "datasets" / fname); !r.empty()) {
                return r;
            }
            if (std::string r = tryResolved(exeDir / ".." / ".." / ".." / "cfg" / "datasets" / fname); !r.empty()) {
                return r;
            }
        }
    }

    return userPath;
}

static std::string trimCopy(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) {
        ++start;
    }
    return s.substr(start);
}

// txt：一行一个类名，第一行是 class 0，顺序要和训练一致
static std::vector<std::string> loadClassNamesFromTextFile(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream in(path);
    if (!in) {
        std::cerr << "无法打开类别文件: " << path << "\n";
        return names;
    }
    std::string line;
    while (std::getline(in, line)) {
        line = trimCopy(line);
        if (!line.empty()) {
            names.push_back(std::move(line));
        }
    }
    return names;
}

// data.yaml 顶层 names:，支持 map(0: cat) 或列表
static std::vector<std::string> loadClassNamesFromUltralyticsDataYaml(const std::string& path) {
    try {
        const YAML::Node root = YAML::LoadFile(path);
        const YAML::Node names = root["names"];
        if (!names || names.IsNull()) {
            std::cerr << "yaml 中未找到顶层 names 字段: " << path << "\n";
            return {};
        }

        if (names.IsMap()) {
            size_t maxIdx = 0;
            for (YAML::const_iterator it = names.begin(); it != names.end(); ++it) {
                const int k = it->first.as<int>();
                if (k >= 0 && static_cast<size_t>(k) > maxIdx) {
                    maxIdx = static_cast<size_t>(k);
                }
            }
            std::vector<std::string> out(maxIdx + 1);
            for (YAML::const_iterator it = names.begin(); it != names.end(); ++it) {
                const int k = it->first.as<int>();
                if (k >= 0 && static_cast<size_t>(k) < out.size()) {
                    out[static_cast<size_t>(k)] = it->second.as<std::string>();
                }
            }
            return out;
        }

        if (names.IsSequence()) {
            std::vector<std::string> out;
            out.reserve(names.size());
            for (std::size_t i = 0; i < names.size(); ++i) {
                out.push_back(names[i].as<std::string>());
            }
            return out;
        }

        std::cerr << "names 类型不支持（需 map 或 sequence）: " << path << "\n";
        return {};
    } catch (const YAML::Exception& e) {
        std::cerr << "yaml-cpp 解析失败: " << e.what() << "\n";
        return {};
    }
}

static bool pathLooksLikeYaml(const std::string& path) {
    const auto dot = path.rfind('.');
    if (dot == std::string::npos) {
        return false;
    }
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".yaml" || ext == ".yml";
}

// --names：yaml 或 txt
static std::vector<std::string> loadClassNamesFromNamesArg(const std::string& path) {
    const std::string resolved = resolveClassNamesInputPath(path);
    if (resolved != path) {
        std::cout << "类别名文件: \"" << path << "\" -> \"" << resolved << "\"\n";
    }
    std::error_code ec;
    if (!fs::exists(resolved, ec)) {
        std::cerr << "找不到类别名文件: " << resolved << "\n"
                  << "（原始参数: " << path << "）\n"
                  << "请使用绝对路径，或相对工程目录的路径，例如: cfg/datasets/coco.yaml\n";
        return {};
    }

    if (pathLooksLikeYaml(resolved)) {
        std::vector<std::string> v = loadClassNamesFromUltralyticsDataYaml(resolved);
        if (v.empty()) {
            std::cerr << "从 yaml 未得到类别名，请确认存在 names: 块（参见 cfg/datasets/coco.yaml）\n";
        } else {
            std::cout << "已加载 " << v.size() << " 个类别名（data yaml）\n";
        }
        return v;
    }
    std::vector<std::string> v = loadClassNamesFromTextFile(resolved);
    if (!v.empty()) {
        std::cout << "已加载 " << v.size() << " 个类别名（文本，每行一类）\n";
    }
    return v;
}

static std::string formatDetectLabel(const std::vector<std::string>& classNames, size_t classIndex, float score) {
    if (classIndex < classNames.size() && !classNames[classIndex].empty()) {
        return std::format("{} {:.2f}", classNames[classIndex], score);
    }
    return std::format("cls{} {:.2f}", classIndex, score);
}

// 可缩放窗口，大小跟图一致（和分割里那几个窗口行为一致）
static void imshowSized(const char* windowName, const cv::Mat& img) {
    if (img.empty()) {
        return;
    }
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, img);
    cv::resizeWindow(windowName, img.cols, img.rows);
}

static void printUsage(const char* prog) {
    std::cerr
        << "用法：\n"
        << "  分割\n"
        << "    " << prog << " <model.onnx> [--names <data.yaml|classes.txt>] <图片路径>\n"
        << "    " << prog << " <model.onnx> [--names <data.yaml|classes.txt>] --camera [摄像头序号]\n"
        << "  检测\n"
        << "    " << prog << " <model.onnx> --detect [--names <data.yaml|classes.txt>] <图片路径>\n"
        << "    " << prog << " <model.onnx> --detect [--names <data.yaml|classes.txt>] --camera [摄像头序号]\n"
        << "  旋转框（OBB）\n"
        << "    " << prog << " <model.onnx> --obb [--names <data.yaml|classes.txt>] <图片路径>\n"
        << "    " << prog << " <model.onnx> --obb [--names <data.yaml|classes.txt>] --camera [摄像头序号]\n"
        << "  姿态\n"
        << "    " << prog << " <model.onnx> --pose [--names <data.yaml|classes.txt>] <图片路径>\n"
        << "    " << prog << " <model.onnx> --pose [--names <data.yaml|classes.txt>] --camera [摄像头序号]\n"
        << "\n"
        << "  --names：yaml 取 data.yaml 中的 names；txt 则每行一个类名。类别数须与模型 nc 一致。\n"
        << "  摄像头预览时按 q 或 ESC 退出。\n";
}

static void visualizeSingleImage(const cv::Mat& image, const std::vector<YOLOInferResult>& results,
                                 const std::vector<std::string>& classNames) {
    for (auto& res : results) {
        cv::Mat mask = res.mask.clone();
        cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
        cv::rectangle(mask, res.rect, cv::Scalar(255, 255, 255), 1);
        const std::string label = formatDetectLabel(classNames, res.classIndex, res.score);
        std::string win_name = std::format("mask_{}", label);
        imshowSized(win_name.c_str(), mask);
    }

    imshowSized("original", image);
    cv::waitKey();
    cv::destroyAllWindows();
}

static void visualizeCameraFrame(cv::Mat& frame, const std::vector<YOLOInferResult>& results,
                                 const std::vector<std::string>& classNames) {
    static const cv::Scalar kPalette[] = {
        {0, 255, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 255, 255},
        {255, 0, 255},
    };
    constexpr size_t kPaletteN = sizeof(kPalette) / sizeof(kPalette[0]);

    cv::Mat vis = frame.clone();
    for (const auto& res : results) {
        const cv::Scalar& c = kPalette[res.classIndex % kPaletteN];
        if (!res.mask.empty()) {
            cv::Mat colored(vis.size(), vis.type(), cv::Scalar::all(0));
            colored.setTo(c, res.mask);
            cv::addWeighted(vis, 1.0, colored, 0.35, 0, vis);
        }
        cv::rectangle(vis, res.rect, c, 2);
        std::string label = formatDetectLabel(classNames, res.classIndex, res.score);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int ty = std::max(ts.height, res.rect.y - 4);
        cv::putText(vis, label, {res.rect.x, ty}, cv::FONT_HERSHEY_SIMPLEX, 0.5, c, 1);
    }
    imshowSized("YOLO26 Seg (camera)", vis);
}

static const std::pair<int, int> kCoco17Skeleton[] = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10},
    {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16},
};

static void drawPoseOverlay(cv::Mat& vis, const PoseInferResult& res,
                            const std::vector<std::string>& classNames) {
    const cv::Scalar boxColor(0, 255, 0);
    cv::rectangle(vis, res.rect, boxColor, 2);
    std::string label = std::format("pose {}", formatDetectLabel(classNames, res.classIndex, res.score));
    cv::putText(vis, label, {res.rect.x, std::max(0, res.rect.y - 6)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, boxColor, 1);

    const auto& kpts = res.keypoints;
    for (size_t i = 0; i < kpts.size(); i++) {
        const auto& p = kpts[i];
        if (p.conf < 0.25f) {
            continue;
        }
        cv::circle(vis, {static_cast<int>(p.x), static_cast<int>(p.y)}, 3, cv::Scalar(0, 0, 255), -1);
    }

    if (kpts.size() == 17) {
        for (const auto& [a, b] : kCoco17Skeleton) {
            if (a >= static_cast<int>(kpts.size()) || b >= static_cast<int>(kpts.size())) {
                continue;
            }
            if (kpts[static_cast<size_t>(a)].conf < 0.25f || kpts[static_cast<size_t>(b)].conf < 0.25f) {
                continue;
            }
            cv::line(vis, {static_cast<int>(kpts[static_cast<size_t>(a)].x), static_cast<int>(kpts[static_cast<size_t>(a)].y)},
                     {static_cast<int>(kpts[static_cast<size_t>(b)].x), static_cast<int>(kpts[static_cast<size_t>(b)].y)},
                     cv::Scalar(255, 128, 0), 2);
        }
    }
}

static void visualizePoseSingleImage(const cv::Mat& image, const std::vector<PoseInferResult>& results,
                                     const std::vector<std::string>& classNames) {
    cv::Mat vis = image.clone();
    for (const auto& res : results) {
        drawPoseOverlay(vis, res, classNames);
    }
    imshowSized("YOLO26 Pose", vis);
    cv::waitKey();
    cv::destroyAllWindows();
}

static void visualizePoseCameraFrame(cv::Mat& frame, const std::vector<PoseInferResult>& results,
                                     const std::vector<std::string>& classNames) {
    cv::Mat vis = frame.clone();
    for (const auto& res : results) {
        drawPoseOverlay(vis, res, classNames);
    }
    imshowSized("YOLO26 Pose (camera)", vis);
}

static void visualizeDetectSingleImage(const cv::Mat& image, const std::vector<DetectInferResult>& results,
                                       const std::vector<std::string>& classNames) {
    static const cv::Scalar kPalette[] = {
        {0, 255, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 255, 255},
        {255, 0, 255},
    };
    constexpr size_t kPaletteN = sizeof(kPalette) / sizeof(kPalette[0]);

    cv::Mat vis = image.clone();
    for (const auto& res : results) {
        const cv::Scalar& c = kPalette[res.classIndex % kPaletteN];
        cv::rectangle(vis, res.rect, c, 2);
        std::string label = formatDetectLabel(classNames, res.classIndex, res.score);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        int ty = std::max(ts.height, res.rect.y - 4);
        cv::putText(vis, label, {res.rect.x, ty}, cv::FONT_HERSHEY_SIMPLEX, 0.6, c, 1);
    }
    imshowSized("YOLO26 Detect", vis);
    cv::waitKey();
    cv::destroyAllWindows();
}

static void visualizeDetectCameraFrame(cv::Mat& frame, const std::vector<DetectInferResult>& results,
                                       const std::vector<std::string>& classNames) {
    static const cv::Scalar kPalette[] = {
        {0, 255, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 255, 255},
        {255, 0, 255},
    };
    constexpr size_t kPaletteN = sizeof(kPalette) / sizeof(kPalette[0]);

    cv::Mat vis = frame.clone();
    for (const auto& res : results) {
        const cv::Scalar& c = kPalette[res.classIndex % kPaletteN];
        cv::rectangle(vis, res.rect, c, 2);
        std::string label = formatDetectLabel(classNames, res.classIndex, res.score);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        int ty = std::max(ts.height, res.rect.y - 4);
        cv::putText(vis, label, {res.rect.x, ty}, cv::FONT_HERSHEY_SIMPLEX, 0.6, c, 1);
    }
    imshowSized("YOLO26 Detect (camera)", vis);
}

static void drawObbOverlay(cv::Mat& vis, const ObbInferResult& res, const cv::Scalar& c,
                           const std::vector<std::string>& classNames) {
    cv::Point2f pts[4];
    res.rrect.points(pts);
    for (int j = 0; j < 4; j++) {
        cv::line(vis, pts[j], pts[(j + 1) % 4], c, 2);
    }
    const cv::Point2f cpt = res.rrect.center;
    std::string label = std::format("obb {}", formatDetectLabel(classNames, res.classIndex, res.score));
    int baseline = 0;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
    const int tx = static_cast<int>(cpt.x) - ts.width / 2;
    const int ty = std::max(ts.height, static_cast<int>(cpt.y) - 6);
    cv::putText(vis, label, {tx, ty}, cv::FONT_HERSHEY_SIMPLEX, 0.6, c, 1);
}

static void visualizeObbSingleImage(const cv::Mat& image, const std::vector<ObbInferResult>& results,
                                    const std::vector<std::string>& classNames) {
    static const cv::Scalar kPalette[] = {
        {0, 255, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 255, 255},
        {255, 0, 255},
    };
    constexpr size_t kPaletteN = sizeof(kPalette) / sizeof(kPalette[0]);

    cv::Mat vis = image.clone();
    for (const auto& res : results) {
        const cv::Scalar& c = kPalette[res.classIndex % kPaletteN];
        drawObbOverlay(vis, res, c, classNames);
    }
    imshowSized("YOLO26 OBB", vis);
    cv::waitKey();
    cv::destroyAllWindows();
}

static void visualizeObbCameraFrame(cv::Mat& frame, const std::vector<ObbInferResult>& results,
                                    const std::vector<std::string>& classNames) {
    static const cv::Scalar kPalette[] = {
        {0, 255, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 255, 255},
        {255, 0, 255},
    };
    constexpr size_t kPaletteN = sizeof(kPalette) / sizeof(kPalette[0]);

    cv::Mat vis = frame.clone();
    for (const auto& res : results) {
        const cv::Scalar& c = kPalette[res.classIndex % kPaletteN];
        drawObbOverlay(vis, res, c, classNames);
    }
    imshowSized("YOLO26 OBB (camera)", vis);
}

static int runObbSingleImage(Yolo26Obb& model, const std::string& imagePath,
                             const std::vector<std::string>& classNames) {
    const std::string resolved = resolveImageInputPath(imagePath);
    if (resolved != imagePath) {
        std::cout << "图像路径: \"" << imagePath << "\" -> \"" << resolved << "\"\n";
    }
    cv::Mat image = cv::imread(resolved, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << resolved << "（原始参数: " << imagePath << "）\n";
        return 1;
    }

    std::vector<ObbInferResult> results = model.inference(image);
    visualizeObbSingleImage(image, results, classNames);
    return 0;
}

static int runObbCamera(Yolo26Obb& model, int cameraIndex, const std::vector<std::string>& classNames) {
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头 index=" << cameraIndex << "\n";
        return 1;
    }

    cv::Mat frame;
    for (;;) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "读取摄像头帧失败，退出。\n";
            break;
        }

        std::vector<ObbInferResult> results = model.inference(frame);
        visualizeObbCameraFrame(frame, results, classNames);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

static int runDetectSingleImage(Yolo26Detect& model, const std::string& imagePath,
                               const std::vector<std::string>& classNames) {
    const std::string resolved = resolveImageInputPath(imagePath);
    if (resolved != imagePath) {
        std::cout << "图像路径: \"" << imagePath << "\" -> \"" << resolved << "\"\n";
    }
    cv::Mat image = cv::imread(resolved, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << resolved << "（原始参数: " << imagePath << "）\n";
        return 1;
    }

    std::vector<DetectInferResult> results = model.inference(image);
    visualizeDetectSingleImage(image, results, classNames);
    return 0;
}

static int runDetectCamera(Yolo26Detect& model, int cameraIndex, const std::vector<std::string>& classNames) {
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头 index=" << cameraIndex << "\n";
        return 1;
    }

    cv::Mat frame;
    for (;;) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "读取摄像头帧失败，退出。\n";
            break;
        }

        std::vector<DetectInferResult> results = model.inference(frame);
        visualizeDetectCameraFrame(frame, results, classNames);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

static int runSingleImage(Yolo26Seg& model, const std::string& imagePath,
                          const std::vector<std::string>& classNames) {
    const std::string resolved = resolveImageInputPath(imagePath);
    if (resolved != imagePath) {
        std::cout << "图像路径: \"" << imagePath << "\" -> \"" << resolved << "\"\n";
    }
    cv::Mat image = cv::imread(resolved, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << resolved << "（原始参数: " << imagePath << "）\n";
        return 1;
    }

    std::vector<YOLOInferResult> results = model.inference(image);
    visualizeSingleImage(image, results, classNames);
    return 0;
}

static int runCamera(Yolo26Seg& model, int cameraIndex, const std::vector<std::string>& classNames) {
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头 index=" << cameraIndex << "\n";
        return 1;
    }

    cv::Mat frame;
    for (;;) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "读取摄像头帧失败，退出。\n";
            break;
        }

        std::vector<YOLOInferResult> results = model.inference(frame);
        visualizeCameraFrame(frame, results, classNames);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

static int runPoseSingleImage(Yolo26Pose& model, const std::string& imagePath,
                              const std::vector<std::string>& classNames) {
    const std::string resolved = resolveImageInputPath(imagePath);
    if (resolved != imagePath) {
        std::cout << "图像路径: \"" << imagePath << "\" -> \"" << resolved << "\"\n";
    }
    cv::Mat image = cv::imread(resolved, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << resolved << "（原始参数: " << imagePath << "）\n";
        return 1;
    }

    std::vector<PoseInferResult> results = model.inference(image);
    visualizePoseSingleImage(image, results, classNames);
    return 0;
}

static int runPoseCamera(Yolo26Pose& model, int cameraIndex, const std::vector<std::string>& classNames) {
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头 index=" << cameraIndex << "\n";
        return 1;
    }

    cv::Mat frame;
    for (;;) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "读取摄像头帧失败，退出。\n";
            break;
        }

        std::vector<PoseInferResult> results = model.inference(frame);
        visualizePoseCameraFrame(frame, results, classNames);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
/*
* 运行方式:
*  分割:  exe  <model.onnx>  [--names coco.yaml|classes.txt]  <image_path>
*         exe  <model.onnx>  [--names ...]  --camera [index]
*  检测:  exe  <model.onnx>  --detect  [--names coco.yaml|classes.txt]  <image_path>
*         exe  <model.onnx>  --detect  [--names ...]  --camera [index]
*  OBB:   exe  <model.onnx>  --obb  [--names ...]  <image_path>
*         exe  <model.onnx>  --obb  [--names ...]  --camera [index]
*  姿态:  exe  <model.onnx>  --pose  [--names ...]  <image_path>
*         exe  <model.onnx>  --pose  [--names ...]  --camera [index]
*/
int main(int argc, char** argv) {
    initConsoleUtf8();

#ifdef _DEBUG
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
#endif

    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string onnxPath = argv[1];
    const std::string arg2 = argv[2];

    if (arg2 == "--detect") {
        DetectConfig dcfg{
            .modelFile = onnxPath,
            .scoreThreshold = 0.5f,
            .maxDetections = 50,
            // 默认 80 类；自己训的数据改 yaml 里 nc
        };
        Yolo26Detect detModel(dcfg);
        if (!detModel.init()) {
            std::cerr << "检测模型初始化失败。\n";
            return 1;
        }

        std::vector<std::string> classNames;
        int ai = 3;
        while (ai + 1 < argc && std::string(argv[ai]) == "--names") {
            classNames = loadClassNamesFromNamesArg(argv[ai + 1]);
            ai += 2;
        }
        if (ai >= argc) {
            printUsage(argv[0]);
            return 1;
        }
        const std::string nextArg = argv[ai];
        if (nextArg == "--camera" || nextArg == "-c") {
            int camIndex = 0;
            if (ai + 1 < argc) {
                camIndex = std::atoi(argv[ai + 1]);
            }
            return runDetectCamera(detModel, camIndex, classNames);
        }
        return runDetectSingleImage(detModel, nextArg, classNames);
    }

    if (arg2 == "--obb") {
        ObbConfig ocfg{
            .modelFile = onnxPath,
            .scoreThreshold = 0.25f, // 低一点框多，接近 ultralytics 默认
            .maxDetections = 300,   // 和训练 max_det 对齐，别设太小
            // 类数要对数据集（DOTA 常 16，COCO 预训练 80）
            .end2endLayout = true,
            .angleInRadians = true,
        };
        Yolo26Obb obbModel(ocfg);
        if (!obbModel.init()) {
            std::cerr << "旋转框模型初始化失败。\n";
            return 1;
        }

        std::vector<std::string> obbClassNames;
        int ai = 3;
        while (ai + 1 < argc && std::string(argv[ai]) == "--names") {
            obbClassNames = loadClassNamesFromNamesArg(argv[ai + 1]);
            ai += 2;
        }
        if (ai >= argc) {
            printUsage(argv[0]);
            return 1;
        }
        const std::string obbNext = argv[ai];
        if (obbNext == "--camera" || obbNext == "-c") {
            int camIndex = 0;
            if (ai + 1 < argc) {
                camIndex = std::atoi(argv[ai + 1]);
            }
            return runObbCamera(obbModel, camIndex, obbClassNames);
        }
        return runObbSingleImage(obbModel, obbNext, obbClassNames);
    }

    if (arg2 == "--pose") {
        PoseConfig pcfg{
            .modelFile = onnxPath,
            .scoreThreshold = 0.5f,
            .nmsThreshold = 0.5f,
            // 默认 80 类、17 关键点、end2end；老格式导出才用到 nmsThreshold
        };
        Yolo26Pose poseModel(pcfg);
        if (!poseModel.init()) {
            std::cerr << "姿态模型初始化失败。\n";
            return 1;
        }

        std::vector<std::string> poseClassNames;
        int ai = 3;
        while (ai + 1 < argc && std::string(argv[ai]) == "--names") {
            poseClassNames = loadClassNamesFromNamesArg(argv[ai + 1]);
            ai += 2;
        }
        if (ai >= argc) {
            printUsage(argv[0]);
            return 1;
        }
        const std::string poseNext = argv[ai];
        if (poseNext == "--camera" || poseNext == "-c") {
            int camIndex = 0;
            if (ai + 1 < argc) {
                camIndex = std::atoi(argv[ai + 1]);
            }
            return runPoseCamera(poseModel, camIndex, poseClassNames);
        }
        return runPoseSingleImage(poseModel, poseNext, poseClassNames);
    }

    Config config = {
        onnxPath,
        0.5f,
    };

    Yolo26Seg model(config);
    if (!model.init()) {
        std::cerr << "分割模型初始化失败。\n";
        return 1;
    }

    std::vector<std::string> segClassNames;
    int ai = 2;
    while (ai + 1 < argc && std::string(argv[ai]) == "--names") {
        segClassNames = loadClassNamesFromNamesArg(argv[ai + 1]);
        ai += 2;
    }
    if (ai >= argc) {
        printUsage(argv[0]);
        return 1;
    }
    const std::string segNext = argv[ai];
    if (segNext == "--camera" || segNext == "-c") {
        int camIndex = 0;
        if (ai + 1 < argc) {
            camIndex = std::atoi(argv[ai + 1]);
        }
        return runCamera(model, camIndex, segClassNames);
    }

    return runSingleImage(model, segNext, segClassNames);
}
