// src/yolo13_det.cpp
#include "NvInfer.h"
#include "model.h"  // ✅ 包含 model.h，它定义了 Logger
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

// ✅ 使用 model.h 中定义的 Logger 类
Logger gLogger;

int main() {
    auto* builder = nvinfer1::createInferBuilder(gLogger);
    auto* config = builder->createBuilderConfig();

    std::string wts_file = "yolov13n.wts";
    nvinfer1::DataType dt = nvinfer1::DataType::kFLOAT;

#ifdef USE_FP16
    dt = nvinfer1::DataType::kHALF;
#endif

    // ✅ 传入 std::string 类型的 wts_file
    auto* serializedModel = buildEngineYolo13Det(builder, config, dt, wts_file);
    if (!serializedModel) {
        std::cerr << "构建引擎失败！" << std::endl;
        return -1;
    }

    // 保存 engine 文件
    std::ofstream engine_file("yolov13n.engine", std::ios::binary);
    engine_file.write(static_cast<char*>(serializedModel->data()), serializedModel->size());
    engine_file.close();

    std::cout << "✅ 引擎构建成功！已保存为 yolov13n.engine" << std::endl;

    // ✅ 释放资源
    delete serializedModel;  // IHostMemory* 是指针
    delete config;
    delete builder;

    return 0;
}
