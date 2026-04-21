#ifndef YOLO26_CPP_TRTENGINECACHE_H
#define YOLO26_CPP_TRTENGINECACHE_H

#include <string>

#include "NvInfer.h"

namespace trt_engine_cache {

// onnx 同目录、同名 .engine
std::string enginePathFromOnnx(const std::string& onnxPath);

// 没缓存或 onnx 比 engine 新就要重编
bool shouldRebuildEngine(const std::string& onnxPath, const std::string& enginePath);

// 读 .engine，成功才改 engine/context
bool deserializeEngine(const std::string& enginePath, nvinfer1::ILogger& logger, nvinfer1::ICudaEngine*& engine,
                         nvinfer1::IExecutionContext*& context);

// 落盘下次直接加载
bool serializeEngineToFile(nvinfer1::ICudaEngine* engine, const std::string& enginePath);

} // namespace trt_engine_cache

#endif
