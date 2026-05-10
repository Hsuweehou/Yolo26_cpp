#include "yolo26/Yolo26OnnxCpuSession.h"

#include <algorithm>
#include <cstring>
#include <iostream>

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace {

#ifdef _WIN32
std::wstring ToWideString(const std::string& text) {
    if (text.empty()) {
        return {};
    }
    const int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (sizeNeeded <= 0) {
        return {};
    }
    std::wstring wide(static_cast<size_t>(sizeNeeded), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), sizeNeeded);
    if (!wide.empty() && wide.back() == L'\0') {
        wide.pop_back();
    }
    return wide;
}
#endif

nvinfer1::Dims ShapeToTrtDims(const std::vector<int64_t>& shape) {
    nvinfer1::Dims d{};
    const int maxD = nvinfer1::Dims::MAX_DIMS;
    const int n = static_cast<int>(std::min(shape.size(), static_cast<size_t>(maxD)));
    d.nbDims = n;
    for (int i = 0; i < n; ++i) {
        int64_t v = shape[static_cast<size_t>(i)];
        if (v <= 0) {
            v = 1;
        }
        d.d[i] = static_cast<int32_t>(v);
    }
    return d;
}

size_t VolumeFromShape(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (int64_t x : shape) {
        int64_t v = x;
        if (v <= 0) {
            v = 1;
        }
        n *= static_cast<size_t>(v);
    }
    return n;
}

/// 将 ONNX 符号维度（-1）替换为常用静态值：batch=1，4D 空间维默认 640。
std::vector<int64_t> ResolveSymbolicShape(const std::vector<int64_t>& shape) {
    std::vector<int64_t> o = shape;
    for (size_t i = 0; i < o.size(); ++i) {
        if (o[i] < 0) {
            if (i == 0) {
                o[i] = 1;
            } else if (o.size() == 4 && (i == 2 || i == 3)) {
                o[i] = 640;
            } else {
                o[i] = 1;
            }
        }
    }
    return o;
}

} // namespace

namespace yolo26 {

struct Yolo26OnnxCpuSession::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "Yolo26OnnxCpu"};
    Ort::SessionOptions sessionOptions{};
    std::unique_ptr<Ort::Session> session;

    int modelW = 0;
    int modelH = 0;
    size_t inputsNum = 0;
    size_t outputsNum = 0;

    std::vector<std::string> inputNamesStr;
    std::vector<std::string> outputNamesStr;
    std::vector<int64_t> inputShape64;

    std::vector<float> inputHost;
    std::vector<std::vector<float>> outputFloats;

    std::vector<nvinfer1::Dims> vecInputDims;
    std::vector<nvinfer1::Dims> vecOutputDims;
    std::vector<size_t> inputSizes;
    std::vector<size_t> outputSizes;

    Impl() {
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }

    void release() {
        session.reset();
        inputHost.clear();
        outputFloats.clear();
        inputNamesStr.clear();
        outputNamesStr.clear();
        inputShape64.clear();
        vecInputDims.clear();
        vecOutputDims.clear();
        inputSizes.clear();
        outputSizes.clear();
        inputsNum = 0;
        outputsNum = 0;
        modelW = 0;
        modelH = 0;
    }
};

Yolo26OnnxCpuSession::Yolo26OnnxCpuSession() : impl_(std::make_unique<Impl>()) {}

Yolo26OnnxCpuSession::~Yolo26OnnxCpuSession() = default;

bool Yolo26OnnxCpuSession::ready() const {
    return impl_ && impl_->session != nullptr && impl_->inputsNum > 0 && impl_->outputsNum > 0;
}

int32_t Yolo26OnnxCpuSession::deviceId() const {
    return -1;
}

int Yolo26OnnxCpuSession::modelWidth() const {
    return impl_ ? impl_->modelW : 0;
}

int Yolo26OnnxCpuSession::modelHeight() const {
    return impl_ ? impl_->modelH : 0;
}

size_t Yolo26OnnxCpuSession::numInputs() const {
    return impl_ ? impl_->inputsNum : 0;
}

size_t Yolo26OnnxCpuSession::numOutputs() const {
    return impl_ ? impl_->outputsNum : 0;
}

const std::vector<nvinfer1::Dims>& Yolo26OnnxCpuSession::inputDims() const {
    return impl_->vecInputDims;
}

const std::vector<nvinfer1::Dims>& Yolo26OnnxCpuSession::outputDims() const {
    return impl_->vecOutputDims;
}

const std::vector<size_t>& Yolo26OnnxCpuSession::inputSizes() const {
    return impl_->inputSizes;
}

const std::vector<size_t>& Yolo26OnnxCpuSession::outputSizes() const {
    return impl_->outputSizes;
}

float* Yolo26OnnxCpuSession::outputHost(size_t i) {
    if (!impl_ || i >= impl_->outputFloats.size()) {
        return nullptr;
    }
    return impl_->outputFloats[i].data();
}

const float* Yolo26OnnxCpuSession::outputHost(size_t i) const {
    if (!impl_ || i >= impl_->outputFloats.size()) {
        return nullptr;
    }
    return impl_->outputFloats[i].data();
}

bool Yolo26OnnxCpuSession::loadFromOnnx(const std::string& onnxPath, std::string_view /*taskTag*/) {
    impl_->release();

    try {
#ifdef _WIN32
        const std::wstring wpath = ToWideString(onnxPath);
        if (wpath.empty() && !onnxPath.empty()) {
            std::cerr << "Yolo26OnnxCpuSession: UTF-16 path conversion failed\n";
            return false;
        }
        impl_->session = std::make_unique<Ort::Session>(impl_->env, wpath.c_str(), impl_->sessionOptions);
#else
        impl_->session = std::make_unique<Ort::Session>(impl_->env, onnxPath.c_str(), impl_->sessionOptions);
#endif
    } catch (const Ort::Exception& e) {
        std::cerr << "Yolo26OnnxCpuSession: Ort::Exception: " << e.what() << '\n';
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Yolo26OnnxCpuSession: " << e.what() << '\n';
        return false;
    }

    if (!impl_->session) {
        return false;
    }

    const size_t inCount = impl_->session->GetInputCount();
    const size_t outCount = impl_->session->GetOutputCount();
    if (inCount < 1 || outCount < 1) {
        std::cerr << "Yolo26OnnxCpuSession: need at least 1 input and 1 output\n";
        impl_->release();
        return false;
    }
    if (inCount != 1) {
        std::cerr << "Yolo26OnnxCpuSession: only single-input models are supported (got " << inCount << ")\n";
        impl_->release();
        return false;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    impl_->inputNamesStr.reserve(inCount);
    for (size_t i = 0; i < inCount; ++i) {
        auto name = impl_->session->GetInputNameAllocated(i, allocator);
        impl_->inputNamesStr.emplace_back(name.get());
    }
    impl_->outputNamesStr.reserve(outCount);
    for (size_t i = 0; i < outCount; ++i) {
        auto name = impl_->session->GetOutputNameAllocated(i, allocator);
        impl_->outputNamesStr.emplace_back(name.get());
    }

    Ort::TypeInfo inTypeInfo = impl_->session->GetInputTypeInfo(0);
    auto inTensorInfo = inTypeInfo.GetTensorTypeAndShapeInfo();
    if (inTensorInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::cerr << "Yolo26OnnxCpuSession: input must be float32\n";
        impl_->release();
        return false;
    }

    const std::vector<int64_t> inShapeSym = inTensorInfo.GetShape();
    impl_->inputShape64 = ResolveSymbolicShape(inShapeSym);
    if (inShapeSym != impl_->inputShape64) {
        std::cerr << "Yolo26OnnxCpuSession: input shape contained dynamic dims; resolved to [";
        for (size_t i = 0; i < impl_->inputShape64.size(); ++i) {
            std::cerr << impl_->inputShape64[i] << (i + 1 < impl_->inputShape64.size() ? "," : "");
        }
        std::cerr << "] (若与导出尺寸不符请使用静态 ONNX 或调整 ResolveSymbolicShape)\n";
    }

    impl_->vecInputDims.clear();
    impl_->vecInputDims.push_back(ShapeToTrtDims(impl_->inputShape64));
    impl_->inputSizes.clear();
    const size_t inElems = VolumeFromShape(impl_->inputShape64);
    impl_->inputSizes.push_back(inElems * sizeof(float));
    impl_->inputHost.resize(inElems);

    if (impl_->inputShape64.size() >= 4) {
        impl_->modelH = static_cast<int>(impl_->inputShape64[2]);
        impl_->modelW = static_cast<int>(impl_->inputShape64[3]);
    }

    impl_->vecOutputDims.clear();
    impl_->outputSizes.clear();
    impl_->outputFloats.clear();
    impl_->vecOutputDims.reserve(outCount);
    impl_->outputSizes.reserve(outCount);
    impl_->outputFloats.resize(outCount);

    for (size_t oi = 0; oi < outCount; ++oi) {
        Ort::TypeInfo oti = impl_->session->GetOutputTypeInfo(oi);
        auto oTensorInfo = oti.GetTensorTypeAndShapeInfo();
        if (oTensorInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::cerr << "Yolo26OnnxCpuSession: output " << oi << " must be float32\n";
            impl_->release();
            return false;
        }
        const std::vector<int64_t> oshapeSym = oTensorInfo.GetShape();
        const std::vector<int64_t> resolved = ResolveSymbolicShape(oshapeSym);
        if (oshapeSym != resolved) {
            std::cerr << "Yolo26OnnxCpuSession: output " << oi
                      << " had symbolic dims; resolved for buffer sizing (首帧推理后可能再校正)\n";
        }
        impl_->vecOutputDims.push_back(ShapeToTrtDims(resolved));
        const size_t elems = VolumeFromShape(resolved);
        impl_->outputSizes.push_back(elems * sizeof(float));
        impl_->outputFloats[oi].resize(elems);
    }

    impl_->inputsNum = inCount;
    impl_->outputsNum = outCount;
    return true;
}

bool Yolo26OnnxCpuSession::enqueue(const cv::Mat& nchwBlob) {
    if (!ready() || impl_->inputsNum == 0) {
        std::cerr << "Yolo26OnnxCpuSession::enqueue: not ready\n";
        return false;
    }
    if (nchwBlob.empty() || nchwBlob.type() != CV_32F) {
        std::cerr << "Yolo26OnnxCpuSession::enqueue: expect CV_32F NCHW blob\n";
        return false;
    }
    const size_t needBytes = impl_->inputSizes[0];
    const size_t gotBytes = static_cast<size_t>(nchwBlob.total()) * nchwBlob.elemSize();
    if (needBytes != gotBytes) {
        std::cerr << "Yolo26OnnxCpuSession::enqueue: blob bytes mismatch (need " << needBytes << ", got " << gotBytes
                  << ")\n";
        return false;
    }

    std::memcpy(impl_->inputHost.data(), nchwBlob.data, needBytes);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        mem, impl_->inputHost.data(), impl_->inputHost.size(), impl_->inputShape64.data(), impl_->inputShape64.size());

    std::vector<const char*> inNames;
    inNames.push_back(impl_->inputNamesStr[0].c_str());
    std::vector<const char*> outNames;
    outNames.reserve(impl_->outputNamesStr.size());
    for (auto& s : impl_->outputNamesStr) {
        outNames.push_back(s.c_str());
    }

    try {
        auto outs = impl_->session->Run(Ort::RunOptions{nullptr}, inNames.data(), &inputTensor, 1, outNames.data(),
                                        outNames.size());
        if (outs.size() != outNames.size()) {
            std::cerr << "Yolo26OnnxCpuSession: unexpected output count\n";
            return false;
        }
        for (size_t i = 0; i < outs.size(); ++i) {
            const auto outInfo = outs[i].GetTensorTypeAndShapeInfo();
            const size_t ec = outInfo.GetElementCount();
            const float* src = outs[i].GetTensorData<float>();
            if (ec != impl_->outputFloats[i].size()) {
                const std::vector<int64_t> concrete = outInfo.GetShape();
                impl_->outputFloats[i].resize(ec);
                impl_->vecOutputDims[i] = ShapeToTrtDims(ResolveSymbolicShape(concrete));
                impl_->outputSizes[i] = ec * sizeof(float);
            }
            std::memcpy(impl_->outputFloats[i].data(), src, ec * sizeof(float));
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "Yolo26OnnxCpuSession::Run: " << e.what() << '\n';
        return false;
    }
    return true;
}

} // namespace yolo26
