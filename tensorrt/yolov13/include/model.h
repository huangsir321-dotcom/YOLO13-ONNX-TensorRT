// include/model.h
#pragma once

#include "NvInfer.h"
#include <string>
#include <iostream>  // ✅ 必须添加：用于 std::cout 和 std::endl

// 自定义 Logger 类
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 只输出非 INFO 级别的日志（可按需调整）
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;  // ✅ 现在可以正常使用
        }
    }
};

// 构建引擎函数声明
nvinfer1::IHostMemory* buildEngineYolo13Det(
    nvinfer1::IBuilder* builder,
    nvinfer1::IBuilderConfig* config,
    nvinfer1::DataType weight_type,
    const std::string& wts_file
);
