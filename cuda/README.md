# YOLO26 C++/CUDA 结构子目录

- **`Yolo26Graph`（`include/yolo26_cuda/yolo26_graph.hpp` + `src/yolo26_graph.cpp`）**  
  从 `../cfg/models/26/*.yaml` 读取 `backbone` / `head`、按 `scales: n|s|m|l|x` 做 depth/width/max_channels 与 `nc` 等元数据，展平为与 Ultralytics 相同的 **layer 下标**，并做 **NCHW 形状推理**（不含权重、不执行卷积）。

- **`yolo26_graph_print`**  
  可执行文件：打印展平表与输出形状。示例（在 `Yolo26_cpp` 构建树中）：
  `yolo26_graph_print.exe ../../cfg/models/26/yolo26.yaml 640 640`

- **`kernels/elementwise.cu`**  
  示例 CUDA 原语：SiLU 原地核，目标库 `yolo26_cuda_kernels`；后续可在此或新增 `.cu` 中逐步接入 Conv/Pool 等（需自管权重与张量排布）。

结构语义说明见 `include/yolo26_cuda/yolo26_modules.hpp`。
