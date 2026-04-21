#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "TrtEngineCache.h"

namespace fs = std::filesystem;

namespace trt_engine_cache {

std::string enginePathFromOnnx(const std::string& onnxPath) {
    const fs::path p(onnxPath);
    return (p.parent_path() / (p.stem().string() + ".engine")).string();
}

bool shouldRebuildEngine(const std::string& onnxPath, const std::string& enginePath) {
    if (!fs::exists(enginePath)) {
        return true;
    }
    if (!fs::exists(onnxPath)) {
        return false;
    }
    std::error_code ec;
    const auto tOnnx = fs::last_write_time(onnxPath, ec);
    const auto tEng = fs::last_write_time(enginePath, ec);
    if (ec) {
        return true;
    }
    return tOnnx > tEng;
}

bool deserializeEngine(const std::string& enginePath, nvinfer1::ILogger& logger, nvinfer1::ICudaEngine*& engine,
                       nvinfer1::IExecutionContext*& context) {
    engine = nullptr;
    context = nullptr;

    std::ifstream f(enginePath, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return false;
    }
    const auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    if (sz == 0) {
        return false;
    }
    std::vector<char> buf(sz);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    if (!f) {
        return false;
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (runtime == nullptr) {
        return false;
    }
    engine = runtime->deserializeCudaEngine(buf.data(), buf.size());
    delete runtime;

    if (engine == nullptr) {
        return false;
    }
    context = engine->createExecutionContext();
    return context != nullptr;
}

bool serializeEngineToFile(nvinfer1::ICudaEngine* engine, const std::string& enginePath) {
    if (engine == nullptr) {
        return false;
    }
    nvinfer1::IHostMemory* plan = engine->serialize();
    if (plan == nullptr) {
        std::cerr << "TensorRT: engine->serialize() returned null" << std::endl;
        return false;
    }
    std::ofstream ofs(enginePath, std::ios::binary);
    ofs.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    const bool ok = ofs.good();
    delete plan;
    return ok;
}

} // namespace trt_engine_cache
