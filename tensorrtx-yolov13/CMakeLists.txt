cmake_minimum_required(VERSION 3.10)

project(yolov13_camera)

# 基础配置
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# CUDA配置
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
enable_language(CUDA)

# 包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/plugin
)

# 第三方库路径配置
include_directories(/usr/local/cuda-11.8/include)
link_directories(/usr/local/cuda-11.8/lib64)

# TensorRT配置
include_directories(/home/bpz/TensorRT-8.6.1.6/include)
link_directories(/home/bpz/TensorRT-8.6.1.6/lib)

# 编译YOLO插件库
add_library(myplugins SHARED 
    ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu
)
target_link_libraries(myplugins 
    nvinfer 
    cudart
)

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 主程序源文件
set(MAIN_SOURCE 
    ${PROJECT_SOURCE_DIR}/yolov13enginecam.cpp
)

include_directories(include)

# 其他源文件
set(OTHER_SOURCES
    ${PROJECT_SOURCE_DIR}/src/preprocess.cu
    ${PROJECT_SOURCE_DIR}/src/postprocess.cu
    ${PROJECT_SOURCE_DIR}/src/postprocess.cpp
    ${PROJECT_SOURCE_DIR}/src/model.cpp
    ${PROJECT_SOURCE_DIR}/src/block.cpp
    ${PROJECT_SOURCE_DIR}/src/calibrator.cpp
    # ${PROJECT_SOURCE_DIR}/src/utils.cpp
)

# 添加可执行文件
add_executable(yolov13_camera
    ${MAIN_SOURCE}
    ${OTHER_SOURCES}
)

# 链接库
target_link_libraries(yolov13_camera
    myplugins
    nvinfer
    cudart
    ${OpenCV_LIBS}
)

# CUDA编译选项
set_target_properties(yolov13_camera PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)

# 安装规则（可选）
install(TARGETS yolov13_camera
    RUNTIME DESTINATION bin
)

# 输出信息
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

