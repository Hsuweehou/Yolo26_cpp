#include "yolo26/Yolo26BackendFactory.h"

#include <iostream>

#include "yolo26/Yolo26TrtSession.h"
#if defined(YOLO26_HAS_ONNXRUNTIME)
#include "yolo26/Yolo26OnnxCpuSession.h"
#endif

namespace yolo26 {

std::unique_ptr<Yolo26VariantBackend> CreateYolo26VariantBackend(Yolo26BackendKind kind) {
    switch (kind) {
    case Yolo26BackendKind::kTensorRT:
        return std::make_unique<Yolo26TrtSession>();
    case Yolo26BackendKind::kOnnxRuntime:
#if defined(YOLO26_HAS_ONNXRUNTIME)
        return std::make_unique<Yolo26OnnxCpuSession>();
#else
        std::cerr << "yolo26: ONNX Runtime backend not built (define YOLO26_HAS_ONNXRUNTIME / set ONNXRUNTIME_DIR).\n";
        return nullptr;
#endif
    case Yolo26BackendKind::kOpenVINO:
        std::cerr << "yolo26: CreateYolo26VariantBackend: OpenVINO not implemented yet.\n";
        return nullptr;
    default:
        return nullptr;
    }
}

} // namespace yolo26
