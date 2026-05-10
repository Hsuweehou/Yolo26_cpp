# Yolo26_cpp

基于 **ONNX** 的 YOLO26 推理示例：默认使用 **TensorRT 10+（GPU）**，可选 **ONNX Runtime（CPU）**。支持实例分割、目标检测、姿态估计、旋转框（OBB）。单可执行文件 `Yolo26_cpp`，通过命令行选择任务类型与推理后端。

## 功能概览

| 模式 | 开关 | 说明 |
|------|------|------|
| 分割 | 默认（无 `--detect` 等） | 两路输出：检测 + mask 原型 |
| 检测 | `--detect` | 端到端输出，letterbox 后处理 |
| 旋转框 | `--obb` | 旋转框 + letterbox |
| 姿态 | `--pose` | 关键点；支持端到端或旧格式（NMS） |

| 推理后端 | 说明 |
|----------|------|
| **TensorRT**（默认） | 首次运行将引擎缓存为与 ONNX **同目录、同主文件名** 的 `.engine`；ONNX 比 `.engine` 新时会自动重编。需 CUDA + TensorRT。 |
| **ONNX Runtime CPU** | 命令行加 `--cpu`；需 CMake 能发现 `onnxruntime`（见下文 `ONNXRUNTIME_DIR`）。 |

类别名通过 `--names` 指定：Ultralytics 的 `data.yaml`（含 `names:`）或纯文本（每行一个类名）。类别数需与导出模型的 `nc` 一致。

## 依赖

- **CMake** ≥ 3.21，**C++20**
- **CUDA**、**CUDAToolkit**（与 TensorRT 版本匹配）
- **TensorRT 10.x**（Windows 下需 `nvinfer_10.lib` 等；`Yolo26TrtSession` 使用 `enqueueV3`，需 TRT 10+）
- **OpenCV**（含 `opencv2/dnn` 等）
- **yaml-cpp**（解析 `data.yaml` 中的 `names`）
- **ONNX Runtime**（可选，仅在使用 `--cpu` 时需要成功链接；官方 Windows 包含 `include/`、`lib/onnxruntime.lib`、运行目录下的 `onnxruntime.dll` 等）

Windows 下 CMake 会尝试把 TensorRT、OpenCV、yaml-cpp、ONNX Runtime（若启用）的运行时 DLL 复制到 exe 输出目录（并配合 `cmake/copy_dlls_if_missing.ps1`）。

## 路径变量（CMake）

| 变量 | 含义 |
|------|------|
| `THIRD_PARTY_LIBRARY_DIR` | 三方库根目录；未设置时默认为仓库根上级目录下的 `../3rdParty` |
| `OpenCV_DIR` | OpenCV 的 CMake 包路径；Windows 未设置时默认 `THIRD_PARTY_LIBRARY_DIR/OpenCV4.X_GPU/x64/vc16/lib` |
| `TRT_DIR` | TensorRT 根目录（含 `include/`、`lib/`），未设置时默认示例为 `D:/software/TensorRT-10.9.0.34` |
| `YAML_CPP_ROOT` | yaml-cpp 根目录，默认 `THIRD_PARTY_LIBRARY_DIR/yaml-cpp-0.9.0` |
| `ONNXRUNTIME_DIR` | ONNX Runtime 根目录（`include/onnxruntime_cxx_api.h`、`lib/onnxruntime.lib` 等）。未设置时默认同上表下的 `OnnxRuntime/onnxruntime-x64-cpu-1.23.2`（可按本机实际版本修改目录名）；也可设环境变量 `ONNXRUNTIME_DIR`。 |

若未在 `ONNXRUNTIME_DIR` 下找到库，将 **关闭 CPU 后端**（仅 TensorRT），命令行使用 `--cpu` 时运行会报错提示需重新配置。

## 编译示例

```text
cd algorithm/Yolo26_cpp
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

可显式指定 TensorRT / yaml-cpp / ONNX Runtime：

```text
cmake -B build -DTRT_DIR="D:/software/TensorRT-10.9.0.34" ^
  -DYAML_CPP_ROOT="D:/path/to/yaml-cpp" ^
  -DONNXRUNTIME_DIR="D:/path/to/onnxruntime-x64-cpu-1.x.x"
```

生成物一般在 `build/Release/Yolo26_cpp.exe`（多配置 MSVC）或 `build/Yolo26_cpp`（Ninja 等）。

Linux 下需自行安装依赖；`CMakeLists.txt` 中非 Windows 分支的链接库名（如 `nvinfer`、`yaml-cpp`）与 Windows 不同，请按环境调整。

## 运行与参数

在可执行文件所在目录执行时，程序会尝试解析 `cfg/datasets/` 下的类别文件与图片路径（便于从 `build/Release` 直接写相对路径）。

| 参数 | 说明 |
|------|------|
| `--cpu` | 使用 ONNX Runtime 在 **CPU** 上推理；未加则使用 **TensorRT + GPU**（并生成/加载 `.engine`）。 |
| `--names <yaml\|txt>` | 类别名来源；`yaml` 读 `names:`，`.txt` 每行一个类名。 |
| `--camera` / `-c` | 摄像头序号，缺省为 `0`；按 **q** 或 **ESC** 退出。摄像头预览时左上角会叠画 **FPS**（采帧 + 推理 的平滑帧率）。 |

**用法简表**（`--cpu` 可放在命令行任意位置，与 `--names` 可互换顺序）：

```text
  分割    Yolo26_cpp <model.onnx> [--cpu] [--names <data.yaml|classes.txt>] <图片>
          Yolo26_cpp <model.onnx> [--cpu] [--names ...] --camera [序号]
  检测    Yolo26_cpp <model.onnx> --detect [--cpu] [--names ...] <图片|摄像头>
  OBB     Yolo26_cpp <model.onnx> --obb [--cpu] [--names ...] <图片|摄像头>
  姿态    Yolo26_cpp <model.onnx> --pose [--cpu] [--names ...] <图片|摄像头>
```

示例：

```text
Yolo26_cpp.exe yolo26-seg.onnx --names ../../cfg/datasets/coco8-seg.yaml ../../cfg/datasets/bus.jpg
Yolo26_cpp.exe yolo26-detect.onnx --detect --cpu --names ../../cfg/datasets/coco.yaml sample.jpg
Yolo26_cpp.exe yolo26-pose.onnx --pose --names ../../cfg/datasets/coco-pose.yaml --camera 0 --cpu
```

在自有代码中切换后端：将对应配置的 `backendKind` 设为 `yolo26::Yolo26BackendKind::kOnnxRuntime` 或 `kTensorRT`（与命令行默认/TRT、或 `--cpu`/ORT 一致）。

## 工程结构（节选）

```text
Yolo26_cpp/
├── CMakeLists.txt
├── cmake/                  # 如 copy_dlls_if_missing.ps1
├── include/yolo26/         # 对外头文件（变体、后端接口、Trt/ORT 会话等）
├── src/
│   ├── main.cpp            # 命令行、可视化、摄像头 FPS、路径解析
│   ├── common/TrtEngineCache.cpp
│   ├── trt/                # TensorRT 会话、后端工厂
│   ├── cpu/                # Yolo26OnnxCpuSession（ORT CPU，可选编译）
│   └── variants/           # Yolo26Seg / Detect / Pose / Obb
└── cfg/                    # 若仓库内提供，作演示用（非必须）
```

## 说明

- 模型需为与任务对应的 **Ultralytics YOLO26 导出 ONNX**；输出维度和后处理逻辑需匹配各 `Yolo26*.cpp` 中的假设（详见头文件与源码注释）。
- 若初始化失败，终端会提示「模型初始化失败」；使用 `--cpu` 但当前构建未链接 ONNX Runtime 时，会提示需配置 `ONNXRUNTIME_DIR` 并重新 CMake。
