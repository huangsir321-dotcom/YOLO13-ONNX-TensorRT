// src/model.cpp
#include "model.h"
#include "config.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <cassert>
#include <map>
// 权重文件加载工具函数
std::map<std::string, Weights> loadWeights(const std::string& file) {
    std::map<std::string, Weights> weightMap;

    std::ifstream input(file);
    if (!input) {
        std::cerr << "无法打开权重文件: " << file << std::endl;
        return weightMap;
    }

    int32_t count;
    input >> count;
    if (count <= 0) {
        std::cerr << "无效的权重数量: " << count << std::endl;
        return weightMap;
    }

    assert(input.good());

    std::string line;
    while (getline(input, line)) {
        std::stringstream ss(line);
        std::string name;
        ss >> name;
        if (name.empty()) continue;

        Weights wts;
        wts.type = nvinfer1::DataType::kFLOAT;
        wts.count = 0;
        wts.values = nullptr;

        uint32_t size;
        ss >> std::dec >> size;
        if (size <= 0) continue;

        float* val = reinterpret_cast<float*>(malloc(sizeof(float) * size));
        uint32_t div = 32;
        int32_t i = 0;
        while (i < size) {
            input >> std::hex >> val[i];
            ++i;
        }
        wts.values = val;
        wts.count = size;
        weightMap[name] = wts;
    }

    return weightMap;
}

// 添加卷积层（Conv + BN + SiLU）
nvinfer1::IScaleLayer* convBnSilu(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, Weights>& weightMap,
                                  nvinfer1::ITensor& input,
                                  int outch,
                                  int ksize,
                                  int s,
                                  int g,
                                  const std::string& lname) {
    auto* conv = network->addConvolutionNd(input, outch,
                                           nvinfer1::DimsHW{ksize, ksize},
                                           weightMap[lname + ".conv.weight"],
                                           weightMap[lname + ".bn.weight"]);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setPaddingNd(nvinfer1::DimsHW{ksize / 2, ksize / 2});
    conv->setNbGroups(g);
    conv->setPrecision(nvinfer1::DataType::kFLOAT);

    auto* scale = network->addScaleNd(*conv->getOutput(0),
                                      nvinfer1::ScaleMode::kUNIFORM,
                                      weightMap[lname + ".bn.bias"].values,
                                      weightMap[lname + ".bn.weight"].values,
                                      weightMap[lname + ".bn.running_var"].values);
    assert(scale);
    scale->setShift(nvinfer1::Dims{1, {1}, {0}});
    scale->setPower(nvinfer1::Dims{1, {1}, {0}});

    auto* silu = network->addActivation(*scale->getOutput(0), nvinfer1::ActivationType::kSWISH);
    assert(silu);

    return silu;
}

// 构建 YOLOv13 检测网络
nvinfer1::IHostMemory* buildEngineYolo13Det(
    nvinfer1::IBuilder* builder,
    nvinfer1::IBuilderConfig* config,
    nvinfer1::DataType weight_type,
    const std::string& wts_file
) {
    // 创建网络定义
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto* network = builder->createNetworkV2(explicitBatch);
    if (!network) {
        std::cerr << "创建网络失败！" << std::endl;
        return nullptr;
    }

    // 加载权重
    auto weightMap = loadWeights(wts_file);
    if (weightMap.empty()) {
        std::cerr << "加载权重失败！" << std::endl;
        return nullptr;
    }

    // 输入层
    auto* data = network->addInput(INPUT_BLOB_NAME, nvinfer1::DataType::kFLOAT,
                                   nvinfer1::Dims4{1, 3, INPUT_H, INPUT_W});
    assert(data);

    // 示例：构建一个简单的 Backbone（你需要根据实际结构修改）
    // 这里以一个 Conv -> BN -> SiLU 为例
    auto* x = convBnSilu(network, weightMap, *data, 16, 3, 2, 1, "backbone.stem.conv");
    x = convBnSilu(network, weightMap, *x->getOutput(0), 32, 3, 2, 1, "backbone.stage1.conv");
    x = convBnSilu(network, weightMap, *x->getOutput(0), 64, 3, 2, 1, "backbone.stage2.conv");

    // 假设输出层（你可能需要接上 Detect Head）
    auto* output = network->addPoolingNd(*x->getOutput(0), nvinfer1::PoolingType::kAVERAGE,
                                         nvinfer1::DimsHW{8, 8});
    assert(output);
    output->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*output->getOutput(0));

    // 设置优化配置
    builder->setMaxBatchSize(BATCH_SIZE);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

#ifdef USE_FP16
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
#endif

    // 构建序列化引擎
    auto* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "构建引擎失败！" << std::endl;
        network->destroy();
        return nullptr;
    }

    // 序列化
    auto* serializedModel = engine->serialize();

    // 清理
    engine->destroy();
    network->destroy();

    // 释放权重内存
    for (auto& mem : weightMap) {
        free(const_cast<void*>(mem.second.values));
    }

    return serializedModel;
}
