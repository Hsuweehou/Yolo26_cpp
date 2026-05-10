#pragma once

/// 与 `cfg/models/26/*.yaml` 中 `module` 名对应的**结构说明**（供文档与后续手写 CUDA 算子分模块对表）。
/// 前向张量形状以 `Yolo26Graph::infer_shapes` 为准，此处仅描述子模块拓扑意图。
///
/// | 模块         | 典型子结构（Ultralytics 实现侧） |
/// |--------------|----------------------------------|
/// | Conv         | Conv2d(含 autopad) → BatchNorm2d → SiLU |
/// | C3k2         | 若干 Bottleneck/小卷积堆叠，C2f/C3k 系变体；stride 1，通道 c1→c2 |
/// | SPPF         | 多个 MaxPool(ksize=k, stride=1) 同尺度级联再 1×1 融合 |
/// | C2PSA        | 通道/空域注意力块（C2f + PSABlock/Attention 变体） |
/// | nn.Upsample  | 最近邻/双线上采样 ×2，通道不变 |
/// | Concat       | 沿 `dim=1` 拼通道，要求各分支 H、W 一致 |
/// | Detect 系    | P3–P(5/6) 多尺度，DFL+分类；Yolo26 为 `end2end` 时的输出布局见 Ultralytics 文档 |
/// | Segment26 等 | 检测头 + mask/proto 支路（与任务 YAML 一致） |
///
namespace y26::cuda_graph::module_docs {
// 头文件仅作命名空间占位，避免空翻译单元；具体结构见 cfg/models/26/ 中 YAML。
}  // namespace y26::cuda_graph::module_docs
