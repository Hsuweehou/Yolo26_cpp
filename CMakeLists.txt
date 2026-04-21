cmake_minimum_required(VERSION 3.21)
project(Yolo26_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

get_filename_component(_Y26_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

if(NOT DEFINED THIRD_PARTY_LIBRARY_DIR)
    if(DEFINED ENV{THIRD_PARTY_LIBRARY_DIR})
        set(THIRD_PARTY_LIBRARY_DIR $ENV{THIRD_PARTY_LIBRARY_DIR} CACHE PATH "Path to 3rd party libraries")
    else()
        set(THIRD_PARTY_LIBRARY_DIR "${_Y26_REPO_ROOT}/../3rdParty" CACHE PATH "Path to 3rd party libraries")
    endif()
else()
    set(THIRD_PARTY_LIBRARY_DIR ${THIRD_PARTY_LIBRARY_DIR} CACHE PATH "Path to 3rd party libraries")
endif()
message(STATUS "THIRD_PARTY_LIBRARY_DIR = ${THIRD_PARTY_LIBRARY_DIR}")

message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}/${CMAKE_BUILD_TYPE})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}/${CMAKE_BUILD_TYPE})
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/${CMAKE_SYSTEM_NAME}/${CMAKE_BUILD_TYPE})

if(NOT DEFINED CMAKE_BUILD_TYPE)
    message(STATUS "CMAKE_BUILD_TYPE is not defined; defaulting to Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "The build type for project.")
endif()

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_SKIP_INSTALL_RPATH FALSE)

if((NOT MSVC) AND (CMAKE_BUILD_TYPE MATCHES Debug))
    add_compile_definitions(_DEBUG)
endif()

if(NOT DEFINED OpenCV_DIR AND WIN32)
    set(OpenCV_DIR ${THIRD_PARTY_LIBRARY_DIR}/OpenCV4.X_GPU/x64/vc16/lib)
endif()
message(STATUS "OpenCV_DIR = ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found")
else()
    message(FATAL_ERROR "Cannot find OpenCV")
endif()

if(NOT DEFINED TRT_DIR)
    set(TRT_DIR "D:/software/TensorRT-10.9.0.34" CACHE PATH "TensorRT root directory (contains include/ and lib/)")
endif()
message(STATUS "TRT_DIR = ${TRT_DIR}")

if(NOT EXISTS "${TRT_DIR}/include/NvInfer.h")
    message(FATAL_ERROR
        "TensorRT headers not found: \"${TRT_DIR}/include/NvInfer.h\"\n"
        "Install TensorRT 10.x for Windows or pass -DTRT_DIR=<path to TensorRT root>.")
endif()

if(WIN32 AND NOT EXISTS "${TRT_DIR}/lib/nvinfer_10.lib")
    message(FATAL_ERROR
        "TensorRT import library not found: \"${TRT_DIR}/lib/nvinfer_10.lib\"\n"
        "Confirm TRT_DIR points to a TensorRT 10.x Windows package.")
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED YAML_CPP_ROOT)
    set(YAML_CPP_ROOT "${THIRD_PARTY_LIBRARY_DIR}/yaml-cpp-0.9.0" CACHE PATH "yaml-cpp root (include/, lib/, bin/)")
endif()
if(NOT EXISTS "${YAML_CPP_ROOT}/include/yaml-cpp/yaml.h")
    message(FATAL_ERROR "yaml-cpp not found. Expected headers at \"${YAML_CPP_ROOT}/include/yaml-cpp/yaml.h\"")
endif()
if(NOT EXISTS "${YAML_CPP_ROOT}/lib/yaml-cpp.lib")
    message(FATAL_ERROR "yaml-cpp import library not found: \"${YAML_CPP_ROOT}/lib/yaml-cpp.lib\"")
endif()
message(STATUS "YAML_CPP_ROOT = ${YAML_CPP_ROOT}")

add_executable(Yolo26_cpp
    main.cpp
    TrtEngineCache.cpp
    Yolo26Seg.cpp
    Yolo26Pose.cpp
    Yolo26Detect.cpp
    Yolo26Obb.cpp
)

target_include_directories(Yolo26_cpp PRIVATE
    ${TRT_DIR}/include
    ${YAML_CPP_ROOT}/include
)

target_link_directories(Yolo26_cpp PRIVATE ${TRT_DIR}/lib)

if(WIN32)
    target_link_libraries(Yolo26_cpp PRIVATE
        ${OpenCV_LIBS}
        CUDA::cudart
        nvinfer_10
        nvonnxparser_10
        nvinfer_plugin_10
        "$<IF:$<CONFIG:Debug>,${YAML_CPP_ROOT}/lib/yaml-cppd.lib,${YAML_CPP_ROOT}/lib/yaml-cpp.lib>"
    )
else()
    target_link_libraries(Yolo26_cpp PRIVATE
        ${OpenCV_LIBS}
        CUDA::cudart
        nvinfer
        nvonnxparser
    )
    find_library(_Y26_YAML_CPP_LIB NAMES yaml-cpp PATHS "${YAML_CPP_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT _Y26_YAML_CPP_LIB)
        message(FATAL_ERROR "yaml-cpp library not found under ${YAML_CPP_ROOT}/lib (non-Windows)")
    endif()
    target_link_libraries(Yolo26_cpp PRIVATE ${_Y26_YAML_CPP_LIB})
endif()

if(WIN32 AND EXISTS "${YAML_CPP_ROOT}/bin")
    add_custom_command(
        TARGET Yolo26_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<IF:$<CONFIG:Debug>,${YAML_CPP_ROOT}/bin/yaml-cppd.dll,${YAML_CPP_ROOT}/bin/yaml-cpp.dll>"
            "$<TARGET_FILE_DIR:Yolo26_cpp>/$<IF:$<CONFIG:Debug>,yaml-cppd.dll,yaml-cpp.dll>"
        COMMENT "[Yolo26_cpp] Copy yaml-cpp runtime DLL"
    )
endif()

# ##########################################################
# 生成 exe 后自动复制依赖 DLL 到输出目录（Windows）
# ##########################################################
if(WIN32)
    # 仅当 exe 目录中尚不存在同名 DLL 时才复制（不覆盖已有文件）
    add_custom_command(
        TARGET Yolo26_cpp POST_BUILD
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_dlls_if_missing.ps1"
            "$<TARGET_FILE_DIR:Yolo26_cpp>"
            $<TARGET_RUNTIME_DLLS:Yolo26_cpp>
        COMMAND_EXPAND_LISTS
        COMMENT "[Yolo26_cpp] Copy runtime DLLs if missing"
    )

    if(EXISTS "${TRT_DIR}/lib")
        file(GLOB _Y26_TRT_DLLS "${TRT_DIR}/lib/*.dll")
        if(_Y26_TRT_DLLS)
            foreach(_dll IN LISTS _Y26_TRT_DLLS)
                get_filename_component(_dll_name ${_dll} NAME)
                add_custom_command(
                    TARGET Yolo26_cpp POST_BUILD
                    COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File
                        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_dlls_if_missing.ps1"
                        "$<TARGET_FILE_DIR:Yolo26_cpp>"
                        "${_dll}"
                    COMMENT "[Yolo26_cpp] Copy TensorRT DLL if missing: ${_dll_name}"
                )
            endforeach()
            list(LENGTH _Y26_TRT_DLLS _y26_trt_dll_count)
            message(STATUS "TensorRT: ${_y26_trt_dll_count} DLL(s) will be copied from ${TRT_DIR}/lib")
        else()
            message(WARNING "No TensorRT DLLs found in ${TRT_DIR}/lib")
        endif()
    endif()

    if(OpenCV_FOUND)
        get_filename_component(_y26_opencv_lib_dir ${OpenCV_DIR} DIRECTORY)
        get_filename_component(_y26_opencv_root ${_y26_opencv_lib_dir} DIRECTORY)
        set(_y26_opencv_bin_paths
            "${_y26_opencv_root}/bin"
            "${_y26_opencv_root}/x64/vc15/bin"
            "${_y26_opencv_root}/x64/vc16/bin"
            "${_y26_opencv_root}/x64/vc17/bin"
        )
        foreach(_bin_dir IN LISTS _y26_opencv_bin_paths)
            if(EXISTS "${_bin_dir}")
                if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                    file(GLOB _y26_opencv_dlls "${_bin_dir}/Debug/*.dll" "${_bin_dir}/*d.dll")
                else()
                    file(GLOB _y26_opencv_dlls "${_bin_dir}/Release/*.dll" "${_bin_dir}/*.dll")
                endif()
                if(_y26_opencv_dlls)
                    foreach(_dll IN LISTS _y26_opencv_dlls)
                        get_filename_component(_dll_name ${_dll} NAME)
                        if(NOT (CMAKE_BUILD_TYPE STREQUAL "Release" AND _dll_name MATCHES "d\\.dll$"))
                            add_custom_command(
                                TARGET Yolo26_cpp POST_BUILD
                                COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File
                                    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_dlls_if_missing.ps1"
                                    "$<TARGET_FILE_DIR:Yolo26_cpp>"
                                    "${_dll}"
                                COMMENT "[Yolo26_cpp] Copy OpenCV DLL if missing: ${_dll_name}"
                            )
                        endif()
                    endforeach()
                    break()
                endif()
            endif()
        endforeach()
    endif()

    message(STATUS "DLL auto-copy configured for Yolo26_cpp")
endif()
