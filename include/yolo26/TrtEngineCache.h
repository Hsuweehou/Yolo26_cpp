#ifndef YOLO26_TRTENGINECACHE_H
#define YOLO26_TRTENGINECACHE_H

#include <string>

#include "NvInfer.h"

namespace trt_engine_cache {

std::string enginePathFromOnnx(const std::string& onnxPath);

bool shouldRebuildEngine(const std::string& onnxPath, const std::string& enginePath);

bool deserializeEngine(const std::string& enginePath, nvinfer1::ILogger& logger, nvinfer1::ICudaEngine*& engine,
                       nvinfer1::IExecutionContext*& context);

bool serializeEngineToFile(nvinfer1::ICudaEngine* engine, const std::string& enginePath);

} // namespace trt_engine_cache

#endif
