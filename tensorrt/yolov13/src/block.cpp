#include "block.h"
#include <assert.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include "config.h"
#include "model.h"
#include "yololayer.h"

// ------------------------- 基础函数实现 -------------------------
std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    std::ifstream input(file);
    if (!input.is_open()) {
        std::cerr << "Error: Unable to load weight file: " << file << std::endl;
        return weightMap;
    }

    int32_t count;
    input >> count;
    if (count <= 0) {
        std::cerr << "Invalid weight map file." << std::endl;
        return weightMap;
    }

    while (count--) {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
        for (uint32_t x = 0; x < size; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                     std::map<std::string, nvinfer1::Weights> weightMap,
                                     nvinfer1::ITensor& input, std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        pval[i] = 1.0;
    }
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, pval, len};

    return network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
}

nvinfer1::ILayer* convBn(nvinfer1::INetworkDefinition* network,
                        std::map<std::string, nvinfer1::Weights> weightMap,
                        nvinfer1::ITensor& input, int ch, int k, int s, 
                        std::string lname, int g) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    
    if (lname.find(".pe") != std::string::npos) {
        nvinfer1::Weights conv_bias = weightMap[lname + ".conv.bias"];
        nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(
            input, ch, nvinfer1::DimsHW{k,k}, 
            weightMap[lname + ".conv.weight"], conv_bias);
        assert(conv);
        conv->setStrideNd(nvinfer1::DimsHW{s,s});
        conv->setPaddingNd(nvinfer1::DimsHW{k/2, k/2});
        conv->setNbGroups(g);
        return addBatchNorm2d(network, weightMap, *conv->getOutput(0), 
                            lname + ".bn", 1e-3);
    } else {
        nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(
            input, ch, nvinfer1::DimsHW{k,k}, 
            weightMap[lname + ".conv.weight"], bias_empty);
        assert(conv);
        conv->setStrideNd(nvinfer1::DimsHW{s,s});
        conv->setPaddingNd(nvinfer1::DimsHW{k/2, k/2});
        conv->setNbGroups(g);
        return addBatchNorm2d(network, weightMap, *conv->getOutput(0), 
                            lname + ".bn", 1e-3);
    }
}

nvinfer1::IElementWiseLayer* convBnSiLU(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights> weightMap,
                                       nvinfer1::ITensor& input, int ch, 
                                       std::vector<int> k, int s, std::string lname) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(
        input, ch, nvinfer1::DimsHW{k[0], k[1]}, 
        weightMap[lname + ".conv.weight"], bias_empty);
    assert(conv);
    
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setPaddingNd(nvinfer1::DimsHW{k[0]/2, k[1]/2});

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), 
                                              lname + ".bn", 1e-3);

    nvinfer1::IActivationLayer* sigmoid = network->addActivation(
        *bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    
    return network->addElementWise(
        *bn->getOutput(0), *sigmoid->getOutput(0), 
        nvinfer1::ElementWiseOperation::kPROD);
}

// [由于篇幅限制，其他函数实现与之前提供的相同，确保包含所有函数]
// 包括：bottleneck, SPPF, DFL, addYoLoLayer, C3k, C3K2, A2C2f, 
// HyperACE, FullPAD_Tunnel, DownsampleConv, C3AH, ABlock, AAttn

// ------------------------- 模块实现 -------------------------
nvinfer1::ILayer* bottleneck(nvinfer1::INetworkDefinition* network,
                            std::map<std::string, nvinfer1::Weights> weightMap,
                            nvinfer1::ITensor& input, int c1, int c2, 
                            bool shortcut, std::vector<int> k1, 
                            std::vector<int> k2, float e, std::string lname) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBnSiLU(network, weightMap, input, c_, k1, 1, lname + ".cv1");
    auto cv2 = convBnSiLU(network, weightMap, *cv1->getOutput(0), c2, k2, 1, lname + ".cv2");

    if (shortcut && c1 == c2) {
        return network->addElementWise(input, *cv2->getOutput(0), 
                                      nvinfer1::ElementWiseOperation::kSUM);
    }
    return cv2;
}

nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights> weightMap,
                                 nvinfer1::ITensor& input, int c1, int c2, 
                                 int k, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBnSiLU(network, weightMap, input, c_, {1,1}, 1, lname + ".cv1");
    
    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k,k});
    pool1->setStrideNd(nvinfer1::DimsHW{1,1});
    pool1->setPaddingNd(nvinfer1::DimsHW{k/2, k/2});
    
    auto pool2 = network->addPoolingNd(*pool1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k,k});
    pool2->setStrideNd(nvinfer1::DimsHW{1,1});
    pool2->setPaddingNd(nvinfer1::DimsHW{k/2, k/2});
    
    auto pool3 = network->addPoolingNd(*pool2->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k,k});
    pool3->setStrideNd(nvinfer1::DimsHW{1,1});
    pool3->setPaddingNd(nvinfer1::DimsHW{k/2, k/2});
    
    nvinfer1::ITensor* concat_inputs[] = {
        cv1->getOutput(0), pool1->getOutput(0), 
        pool2->getOutput(0), pool3->getOutput(0)
    };
    auto cat = network->addConcatenation(concat_inputs, 4);
    
    return convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1,1}, 1, lname + ".cv2");
}

nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network,
                            std::map<std::string, nvinfer1::Weights> weightMap,
                            nvinfer1::ITensor& input, int ch, int grid, 
                            int k, int s, int p, std::string lname) {
    auto shuffle1 = network->addShuffle(input);
    shuffle1->setReshapeDimensions(nvinfer1::Dims4{kBatchSize, 4, 16, grid});
    shuffle1->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    
    auto softmax = network->addSoftMax(*shuffle1->getOutput(0));
    softmax->setAxes(1 << 1);
    
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto conv = network->addConvolutionNd(
        *softmax->getOutput(0), 1, nvinfer1::DimsHW{1,1}, 
        weightMap[lname], bias_empty);
    conv->setStrideNd(nvinfer1::DimsHW{s,s});
    conv->setPaddingNd(nvinfer1::DimsHW{p,p});
    
    auto shuffle2 = network->addShuffle(*conv->getOutput(0));
    shuffle2->setReshapeDimensions(nvinfer1::Dims3{kBatchSize, 4, grid});
    
    return shuffle2;
}

nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition* network,
                                      std::vector<nvinfer1::IConcatenationLayer*> dets,
                                      const int* strides, int num_outputs,
                                      bool is_segmentation, bool is_pose, bool is_obb) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    
    const int netinfo_count = 9;
    const int total_count = netinfo_count + num_outputs;
    std::vector<int> combinedInfo(total_count);
    
    combinedInfo[0] = is_obb ? kObbNumClass : (is_pose ? kPoseNumClass : kNumClass);
    combinedInfo[1] = kNumberOfPoints;
    combinedInfo[2] = kConfThreshKeypoints;
    combinedInfo[3] = is_obb ? kObbInputW : kInputW;
    combinedInfo[4] = is_obb ? kObbInputH : kInputH;
    combinedInfo[5] = kMaxNumOutputBbox;
    combinedInfo[6] = is_segmentation;
    combinedInfo[7] = is_pose;
    combinedInfo[8] = is_obb;
    
    std::copy(strides, strides + num_outputs, combinedInfo.begin() + netinfo_count);
    
    nvinfer1::PluginField pluginField = {
        "combinedInfo", combinedInfo.data(), 
        nvinfer1::PluginFieldType::kINT32, (int32_t)combinedInfo.size()
    };
    
    nvinfer1::PluginFieldCollection pluginFieldCollection;
    pluginFieldCollection.nbFields = 1;
    pluginFieldCollection.fields = &pluginField;
    
    nvinfer1::IPluginV2* pluginObj = creator->createPlugin("yololayer", &pluginFieldCollection);
    
    std::vector<nvinfer1::ITensor*> inputTensors;
    for (auto& det : dets) {
        inputTensors.push_back(det->getOutput(0));
    }
    
    return network->addPluginV2(inputTensors.data(), inputTensors.size(), *pluginObj);
}

nvinfer1::ILayer* C3k(nvinfer1::INetworkDefinition* network,
                     std::map<std::string, nvinfer1::Weights> weightMap,
                     nvinfer1::ITensor& input, int c1, int c2, int n, 
                     bool shortcut, std::vector<int> k1, 
                     std::vector<int> k2, float e, std::string lname) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBnSiLU(network, weightMap, input, c_, {1,1}, 1, lname + ".cv1");
    auto cv2 = convBnSiLU(network, weightMap, input, c_, {1,1}, 1, lname + ".cv2");
    
    nvinfer1::ITensor* y1 = cv1->getOutput(0);
    for (int i = 0; i < n; ++i) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, k1, k2, 1.0, 
                          lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    
    nvinfer1::ITensor* concat_inputs[] = {y1, cv2->getOutput(0)};
    auto cat = network->addConcatenation(concat_inputs, 2);
    
    return convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1,1}, 1, lname + ".cv3");
}

nvinfer1::IElementWiseLayer* C3K2(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights>& weightMap,
                                 nvinfer1::ITensor& input, int c1, int c2, int n, 
                                 bool c3k, bool shortcut, float e, std::string lname) {
    int c_ = (int)((float)c2 * e);
    
    auto conv1 = convBnSiLU(network, weightMap, input, 2 * c_, {1,1}, 1, lname + ".cv1");
    nvinfer1::Dims dims = conv1->getOutput(0)->getDimensions();
    
    auto split1 = network->addSlice(
        *conv1->getOutput(0), nvinfer1::Dims4{0,0,0,0},
        nvinfer1::Dims4{dims.d[0], dims.d[1]/2, dims.d[2], dims.d[3]},
        nvinfer1::Dims4{1,1,1,1});
    
    auto split2 = network->addSlice(
        *conv1->getOutput(0), nvinfer1::Dims4{0,dims.d[1]/2,0,0},
        nvinfer1::Dims4{dims.d[0], dims.d[1]/2, dims.d[2], dims.d[3]},
        nvinfer1::Dims4{1,1,1,1});
    
    nvinfer1::ITensor* concat_inputs[] = {split1->getOutput(0), split2->getOutput(0)};
    auto cat = network->addConcatenation(concat_inputs, 2);
    
    nvinfer1::ITensor* y = split2->getOutput(0);
    for (int i = 0; i < n; ++i) {
        nvinfer1::ILayer* b;
        if (c3k) {
            b = C3k(network, weightMap, *y, c_, c_, 2, shortcut, {3,3}, {3,3}, 0.5,
                   lname + ".m." + std::to_string(i));
        } else {
            b = bottleneck(network, weightMap, *y, c_, c_, shortcut, {3,3}, {3,3}, 0.5,
                          lname + ".m." + std::to_string(i));
        }
        y = b->getOutput(0);
        
        nvinfer1::ITensor* new_concat_inputs[] = {cat->getOutput(0), b->getOutput(0)};
        cat = network->addConcatenation(new_concat_inputs, 2);
    }
    
    return convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1,1}, 1, lname + ".cv2");
}

nvinfer1::ILayer* A2C2f(nvinfer1::INetworkDefinition* network,
                       std::map<std::string, nvinfer1::Weights> weightMap,
                       nvinfer1::ITensor& input, int c1, int c2, int n, 
                       bool a2, int area, bool residual, 
                       float mlp_ratio, float e, int g, 
                       bool shortcut, std::string lname) {
    int c = (int)((float)c2 * e);
    int num_heads = std::max(c / 32, 1) * 2;
    
    auto conv1 = convBnSiLU(network, weightMap, input, c * 2, {1,1}, 1, lname + ".cv1");
    
    if (a2) {
        auto ablock1 = ABlock(network, weightMap, *conv1->getOutput(0), c, num_heads, 
                             mlp_ratio, area, lname + ".m.0.0");
        auto ablock2 = ABlock(network, weightMap, *ablock1->getOutput(0), c, num_heads, 
                             mlp_ratio, area, lname + ".m.0.1");
        auto ablock3 = ABlock(network, weightMap, *ablock2->getOutput(0), c, num_heads, 
                             mlp_ratio, area, lname + ".m.1.0");
        auto ablock4 = ABlock(network, weightMap, *ablock3->getOutput(0), c, num_heads, 
                             mlp_ratio, area, lname + ".m.1.1");
        
        nvinfer1::ITensor* concat_inputs[] = {
            conv1->getOutput(0), 
            ablock2->getOutput(0), 
            ablock4->getOutput(0)
        };
        auto cat = network->addConcatenation(concat_inputs, 3);
        
        return convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1,1}, 1, lname + ".cv2");
    } else {
        auto c3k_ = C3k(network, weightMap, *conv1->getOutput(0), c * 2, c * 2, 2, 
                       shortcut, {3,3}, {3,3}, 0.5, lname + ".m.0");
        
        nvinfer1::ITensor* concat_inputs[] = {conv1->getOutput(0), c3k_->getOutput(0)};
        auto cat = network->addConcatenation(concat_inputs, 2);
        
        return convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1,1}, 1, lname + ".cv2");
    }
}

nvinfer1::ILayer* HyperACE(nvinfer1::INetworkDefinition* network,
                          std::map<std::string, nvinfer1::Weights> weightMap,
                          nvinfer1::ITensor& input, int c1, int c2, std::string lname) {
    int c_ = c1 / 2;
    
    auto cv1 = convBnSiLU(network, weightMap, input, c1, {1,1}, 1, lname + ".cv1");
    auto cv2 = convBnSiLU(network, weightMap, input, c2, {1,1}, 1, lname + ".cv2");
    
    auto m = C3k(network, weightMap, *cv1->getOutput(0), c_, c_, 2, true, {3,3}, {3,3}, 0.5, lname + ".m");
    
    auto downsample = network->addPooling(*cv2->getOutput(0), nvinfer1::PoolingType::kAVERAGE, {2,2});
    downsample->setStrideNd({2,2});
    
    auto upsample = network->addResize(*m->getOutput(0));
    upsample->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    float scales[] = {1.0f, 1.0f, 2.0f, 2.0f};
    upsample->setScales(scales, 4);
    
    auto branch1 = C3AH(network, weightMap, *downsample->getOutput(0), c_, c_, lname + ".branch1");
    auto branch2 = C3AH(network, weightMap, *upsample->getOutput(0), c_, c_, lname + ".branch2");
    
    nvinfer1::ITensor* concat_inputs[] = {branch1->getOutput(0), branch2->getOutput(0)};
    auto cat = network->addConcatenation(concat_inputs, 2);
    
    return convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1,1}, 1, lname + ".conv_out");
}

nvinfer1::ILayer* FullPAD_Tunnel(nvinfer1::INetworkDefinition* network,
                                nvinfer1::ITensor& input) {
    return network->addIdentity(input);
}

nvinfer1::ILayer* DownsampleConv(nvinfer1::INetworkDefinition* network,
                                std::map<std::string, nvinfer1::Weights> weightMap,
                                nvinfer1::ITensor& input, int c1, int c2, std::string lname) {
    auto pool = network->addPooling(input, nvinfer1::PoolingType::kAVERAGE, {2,2});
    pool->setStrideNd({2,2});
    return convBnSiLU(network, weightMap, *pool->getOutput(0), c2, {1,1}, 1, lname + ".channel_adjust");
}

nvinfer1::ILayer* C3AH(nvinfer1::INetworkDefinition* network,
                      std::map<std::string, nvinfer1::Weights> weightMap,
                      nvinfer1::ITensor& input, int c1, int c2, std::string lname) {
    auto cv1 = convBnSiLU(network, weightMap, input, c1, {1,1}, 1, lname + ".cv1");
    auto cv2 = convBnSiLU(network, weightMap, input, c1, {1,1}, 1, lname + ".cv2");
    
    auto hgnn = convBnSiLU(network, weightMap, *cv2->getOutput(0), c1, {1,1}, 1, lname + ".m.hgnn");
    
    nvinfer1::ITensor* concat_inputs[] = {cv1->getOutput(0), hgnn->getOutput(0)};
    auto cat = network->addConcatenation(concat_inputs, 2);
    
    return convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1,1}, 1, lname + ".cv3");
}

nvinfer1::ILayer* ABlock(nvinfer1::INetworkDefinition* network,
                        std::map<std::string, nvinfer1::Weights> weightMap,
                        nvinfer1::ITensor& input, int dim, int num_heads, 
                        float mlp_ratio, int area, std::string lname) {
    int mlp_hidden_dim = (int)(dim * mlp_ratio);
    
    auto attn = AAttn(network, weightMap, input, dim, num_heads, mlp_ratio, area, lname + ".attn");
    auto sum = network->addElementWise(input, *attn->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    
    auto mlp1 = convBnSiLU(network, weightMap, *sum->getOutput(0), mlp_hidden_dim * 2, {1,1}, 1, lname + ".mlp.0");
    auto mlp2 = convBn(network, weightMap, *mlp1->getOutput(0), dim * 2, 1, 1, lname + ".mlp.1");
    
    return network->addElementWise(*sum->getOutput(0), *mlp2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
}

nvinfer1::ILayer* AAttn(nvinfer1::INetworkDefinition* network,
                       std::map<std::string, nvinfer1::Weights> weightMap,
                       nvinfer1::ITensor& input, int dim, int num_heads, 
                       float mlp_ratio, int area, std::string lname) {
    int head_dim = std::max(dim / num_heads, 1);
    int all_head_dim = head_dim * num_heads;
    float scale = 1.0f / sqrt(head_dim);
    
    auto dims = input.getDimensions();
    int B = dims.d[0];
    int H = dims.d[2];
    int W = dims.d[3];
    int N = H * W;
    
    auto qkv = convBn(network, weightMap, input, all_head_dim * 3 * 2, 1, 1, lname + ".qkv");
    
    auto reshape = network->addShuffle(*qkv->getOutput(0));
    reshape->setReshapeDimensions(nvinfer1::Dims3{B, -1, N});
    reshape->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});
    
    if (area > 1) {
        B = B * area;
        N = N / area;
    }
    
    auto reshape1 = network->addShuffle(*reshape->getOutput(0));
    reshape1->setReshapeDimensions(nvinfer1::Dims4{B, N, num_heads, head_dim * 3 * 2});
    reshape1->setSecondTranspose(nvinfer1::Permutation{0, 2, 3, 1});
    
    auto q = network->addSlice(
        *reshape1->getOutput(0), nvinfer1::Dims4{0,0,0,0},
        nvinfer1::Dims4{B, num_heads, head_dim * 2, N},
        nvinfer1::Dims4{1,1,1,1});
    
    auto k = network->addSlice(
        *reshape1->getOutput(0), nvinfer1::Dims4{0,0,head_dim * 2,0},
        nvinfer1::Dims4{B, num_heads, head_dim * 2, N},
        nvinfer1::Dims4{1,1,1,1});
    
    auto v = network->addSlice(
        *reshape1->getOutput(0), nvinfer1::Dims4{0,0,head_dim * 4,0},
        nvinfer1::Dims4{B, num_heads, head_dim * 2, N},
        nvinfer1::Dims4{1,1,1,1});
    
    auto qT = network->addShuffle(*q->getOutput(0));
    qT->setFirstTranspose(nvinfer1::Permutation{0,1,3,2});
    
    auto matmul = network->addMatrixMultiply(
        *qT->getOutput(0), nvinfer1::MatrixOperation::kNONE,
        *k->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    
    float* scale_val = reinterpret_cast<float*>(malloc(sizeof(float)));
    *scale_val = scale;
    nvinfer1::Weights scale_w{nvinfer1::DataType::kFLOAT, scale_val, 1};
    nvinfer1::Weights shift_w{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::Weights power_w{nvinfer1::DataType::kFLOAT, nullptr, 0};
    
    auto scale_layer = network->addScale(
        *matmul->getOutput(0), nvinfer1::ScaleMode::kUNIFORM, 
        shift_w, scale_w, power_w);
    
    auto softmax = network->addSoftMax(*scale_layer->getOutput(0));
    softmax->setAxes(1 << 3);
    
    auto attnT = network->addShuffle(*softmax->getOutput(0));
    attnT->setFirstTranspose(nvinfer1::Permutation{0,1,3,2});
    
    auto matmul1 = network->addMatrixMultiply(
        *v->getOutput(0), nvinfer1::MatrixOperation::kNONE,
        *attnT->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    
    auto transpose4 = network->addShuffle(*matmul1->getOutput(0));
    transpose4->setFirstTranspose(nvinfer1::Permutation{0,3,1,2});
    
    if (area > 1) {
        B = B / area;
        N = N * area;
    }
    
    auto reshape3 = network->addShuffle(*transpose4->getOutput(0));
    reshape3->setReshapeDimensions(nvinfer1::Dims4{B, H, W, -1});
    
    auto transpose6 = network->addShuffle(*reshape3->getOutput(0));
    transpose6->setFirstTranspose(nvinfer1::Permutation{0,3,1,2});
    
    auto transpose5 = network->addShuffle(*v->getOutput(0));
    transpose5->setFirstTranspose(nvinfer1::Permutation{0,3,1,2});
    
    auto reshape4 = network->addShuffle(*transpose5->getOutput(0));
    reshape4->setReshapeDimensions(nvinfer1::Dims4{B, H, W, all_head_dim * 2});
    
    auto transpose7 = network->addShuffle(*reshape4->getOutput(0));
    transpose7->setFirstTranspose(nvinfer1::Permutation{0,3,1,2});
    
    auto pe = convBn(network, weightMap, *transpose7->getOutput(0), all_head_dim * 2, 7, 1, 
                    lname + ".pe", all_head_dim * 2);
    
    auto sum = network->addElementWise(*pe->getOutput(0), *transpose6->getOutput(0), 
                                      nvinfer1::ElementWiseOperation::kSUM);
    
    return convBn(network, weightMap, *sum->getOutput(0), all_head_dim * 2, 1, 1, lname + ".proj");
}

