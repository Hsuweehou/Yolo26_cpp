#ifndef YOLO26_YOLO26BACKENDFACTORY_H
#define YOLO26_YOLO26BACKENDFACTORY_H

// 与 YOLO26_TensorRT_ONNX 中 CreateGpuBackend / CreateCpuBackend 并列

#include <memory>

#include "yolo26/Yolo26VariantBackend.h"

namespace yolo26 {

std::unique_ptr<Yolo26VariantBackend> CreateYolo26VariantBackend(Yolo26BackendKind kind);

} // namespace yolo26

#endif
