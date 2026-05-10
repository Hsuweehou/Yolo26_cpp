#include "yolo26_cuda/yolo26_graph.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: yolo26_graph_print <cfg/models/26/yolo26.yaml> [W=640] [H=640]\n";
        return 1;
    }
    const std::string path = argv[1];
    int w = 640;
    int h = 640;
    if (argc > 2) {
        w = std::stoi(argv[2]);
    }
    if (argc > 3) {
        h = std::stoi(argv[3]);
    }
    const auto g = y26::cuda_graph::Yolo26Graph::fromYamlFile(path, y26::cuda_graph::ScaleLetter::kN, std::nullopt);
    if (!g) {
        std::cerr << "加载或解析失败: " << path << std::endl;
        return 2;
    }
    std::cout << g->dumpTable(w, h, 1) << std::endl;
    return 0;
}
