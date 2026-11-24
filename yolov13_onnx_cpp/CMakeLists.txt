cmake_minimum_required(VERSION 3.10)
project(yolov13_onnx_cpp)

set(CMAKE_CXX_STANDARD 17)

# 查找依赖
find_package(OpenCV REQUIRED)
set(ONNXRUNTIME_DIR "/home/bpz/yolo系列网课/onnxruntime-linux-x64-1.18.1")

# 包含目录
include_directories(
    ${ONNXRUNTIME_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    include
)

# 源文件
set(COMMON_SOURCES
    src/preprocess.cpp
    src/postprocess.cpp
    src/utils.cpp
)

# 主程序（图片检测）
add_executable(yolov13_image src/main.cpp ${COMMON_SOURCES})
target_link_libraries(yolov13_image
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

# 摄像头程序
add_executable(yolov13_cam src/cam.cpp ${COMMON_SOURCES})
target_link_libraries(yolov13_cam
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)
