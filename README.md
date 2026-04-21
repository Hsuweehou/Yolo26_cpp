# Yolo26_cpp

基于 **ONNX → TensorRT** 的 YOLO26 推理示例：实例分割、目标检测、姿态估计、旋转框（OBB）。单可执行文件 `Yolo26_cpp`，命令行选任务类型。

## 功能概览

| 模式 | 开关 | 说明 |
|------|------|------|
| 分割 | 默认（无 `--detect` 等） | 两路输出：检测 + mask 原型 |
| 检测 | `--detect` | 端到端输出，letterbox 后处理 |
| 旋转框 | `--obb` | 旋转框 + letterbox |
| 姿态 | `--pose` | 关键点；支持端到端或旧格式（NMS） |

类别名通过 `--names` 指定：Ultralytics 的 `data.yaml`（含 `names:`）或纯文本（每行一个类名）。类别数需与导出模型的 `nc` 一致。

首次运行会将 TensorRT 引擎缓存为与 ONNX **同目录、同主文件名** 的 `.engine`；ONNX 比 `.engine` 新时会自动重编。

## 依赖

- **CMake** ≥ 3.21，**C++20**
- **CUDA**（与 TensorRT 版本匹配）
- **TensorRT 10.x**（Windows 下需 `nvinfer_10.lib` 等；代码使用 `enqueueV3`，需 TRT 10+）
- **OpenCV**（含 `opencv2/dnn` 等）
- **yaml-cpp**（解析 `data.yaml` 中的 `names`）

Windows 下 CMake 会尝试把 TensorRT / OpenCV / yaml-cpp 的运行时 DLL 复制到 exe 输出目录（见 `cmake/copy_dlls_if_missing.ps1`）。

## 路径变量（CMake）

| 变量 | 含义 |
|------|------|
| `THIRD_PARTY_LIBRARY_DIR` | 三方库根目录；未设置时默认为仓库上级目录下的 `../3rdParty` |
| `OpenCV_DIR` | OpenCV 的 CMake 包路径；Windows 默认指向 `THIRD_PARTY_LIBRARY_DIR/OpenCV4.X_GPU/x64/vc16/lib` |
| `TRT_DIR` | TensorRT 根目录（含 `include/`、`lib/`），默认示例为 `D:/software/TensorRT-10.9.0.34` |
| `YAML_CPP_ROOT` | yaml-cpp 根目录（含 `include/yaml-cpp/`、`lib/`），默认 `THIRD_PARTY_LIBRARY_DIR/yaml-cpp-0.9.0` |

也可通过环境变量 `THIRD_PARTY_LIBRARY_DIR` 传入。

## 编译示例

```bash
cd algorithm/Yolo26_cpp_v1
cmake -B build -DCMAKE_BUILD_TYPE=Release ^
  -DTRT_DIR="D:/software/TensorRT-10.9.0.34" ^
  -DYAML_CPP_ROOT="path/to/yaml-cpp"
cmake --build build --config Release
```

生成物一般在 `build/Release/Yolo26_cpp.exe`（MSVC）或 `build/Yolo26_cpp`（单配置生成器）。

Linux 下需自行安装 TensorRT、OpenCV、yaml-cpp，并设置 `TRT_DIR`、`YAML_CPP_ROOT`、`OpenCV_DIR` 等，链接名在 `CMakeLists.txt` 的 `else()` 分支中。

## 运行示例

在可执行文件所在目录执行时，程序会尝试解析 `cfg/datasets/` 下的类别文件与图片路径（便于从 `build/Release` 直接跑相对路径）。

```text
用法：
  分割
    Yolo26_cpp <model.onnx> [--names <data.yaml|classes.txt>] <图片路径>
    Yolo26_cpp <model.onnx> [--names <data.yaml|classes.txt>] --camera [摄像头序号]
  检测
    Yolo26_cpp <model.onnx> --detect [--names <data.yaml|classes.txt>] <图片路径>
    ...
  旋转框（OBB）
    Yolo26_cpp <model.onnx> --obb [--names ...] <图片路径>
    ...
  姿态
    Yolo26_cpp <model.onnx> --pose [--names ...] <图片路径>
    ...

  --names：yaml 取 data.yaml 中的 names；txt 则每行一个类名。类别数须与模型 nc 一致。
  摄像头预览时按 q 或 ESC 退出。
```

示例：

```bash
Yolo26_cpp.exe yolo26-seg.onnx --names ../../cfg/datasets/coco8-seg.yaml ../../cfg/datasets/bus.jpg
Yolo26_cpp.exe yolo26-detect.onnx --detect --names ../../cfg/datasets/coco.yaml sample.jpg
```

仓库内附带 `cfg/`（含 Ultralytics 风格 yaml），可按需指向自己的 `data.yaml` 或类别 txt。

## 工程结构（节选）

```
Yolo26_cpp_v1/
├── main.cpp              # 命令行、可视化、路径解析
├── TrtEngineCache.*      # .engine 缓存与序列化
├── Yolo26Seg.*           # 分割
├── Yolo26Detect.*        # 检测
├── Yolo26Pose.*          # 姿态
├── Yolo26Obb.*           # 旋转框
└── cfg/                  # 数据集 / 模型参考 yaml（非编译必需，运行演示用）
```

## 说明

- 模型需为与上述任务对应的 **Ultralytics YOLO26 导出 ONNX**；输出维度与后处理逻辑需匹配（详见各 `Yolo26*.h` 注释）。
- 若初始化失败，终端会提示分割 / 检测 / 旋转框 / 姿态中对应的「模型初始化失败」信息。
