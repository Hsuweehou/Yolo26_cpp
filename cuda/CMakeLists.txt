# YOLO26 结构图（YAML → 展平/缩放/形状）与小型 CUDA 算子；由算法/Yolo26_cpp/CMakeLists.txt add_subdirectory 引入。

if(NOT DEFINED YAML_CPP_ROOT)
    message(FATAL_ERROR "cuda/CMakeLists.txt: YAML_CPP_ROOT not set; configure parent Yolo26_cpp first.")
endif()

if(NOT EXISTS "${YAML_CPP_ROOT}/include/yaml-cpp/yaml.h")
    message(FATAL_ERROR "cuda/CMakeLists.txt: yaml-cpp headers not found: ${YAML_CPP_ROOT}")
endif()

add_library(yolo26_cuda_graph STATIC
    src/yolo26_graph.cpp
)
target_include_directories(yolo26_cuda_graph PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${YAML_CPP_ROOT}/include
)
target_compile_features(yolo26_cuda_graph PUBLIC cxx_std_20)

if(MSVC)
    target_link_libraries(yolo26_cuda_graph PRIVATE
        "$<IF:$<CONFIG:Debug>,${YAML_CPP_ROOT}/lib/yaml-cppd.lib,${YAML_CPP_ROOT}/lib/yaml-cpp.lib>"
    )
else()
    find_library(_Y26_CUDA_YAML_LIB NAMES yaml-cpp PATHS "${YAML_CPP_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT _Y26_CUDA_YAML_LIB)
        message(FATAL_ERROR "yaml-cpp library not under ${YAML_CPP_ROOT}/lib")
    endif()
    target_link_libraries(yolo26_cuda_graph PRIVATE ${_Y26_CUDA_YAML_LIB})
endif()

add_executable(yolo26_graph_print
    tools/yolo26_graph_print.cpp
)
target_link_libraries(yolo26_graph_print PRIVATE yolo26_cuda_graph)

if(WIN32 AND EXISTS "${YAML_CPP_ROOT}/bin")
    add_custom_command(
        TARGET yolo26_graph_print POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<IF:$<CONFIG:Debug>,${YAML_CPP_ROOT}/bin/yaml-cppd.dll,${YAML_CPP_ROOT}/bin/yaml-cpp.dll>"
            "$<TARGET_FILE_DIR:yolo26_graph_print>/$<IF:$<CONFIG:Debug>,yaml-cppd.dll,yaml-cpp.dll>"
        COMMENT "[yolo26_graph_print] Copy yaml-cpp runtime DLL"
    )
endif()

# SiLU 等基元核（前向无依赖 yaml）
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
# 若 "native" 在部分环境下失败，可显式 -DCMAKE_CUDA_ARCHITECTURES=75
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
endif()
add_library(yolo26_cuda_kernels STATIC
    kernels/elementwise.cu
)
set_target_properties(yolo26_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(yolo26_cuda_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(yolo26_cuda_kernels PUBLIC CUDA::cudart)
