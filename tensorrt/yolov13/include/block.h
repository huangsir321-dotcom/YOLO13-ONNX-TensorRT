#pragma once
#include "NvInfer.h"
#include <map>
#include <string>
#include <vector>

// 基础函数
std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);
nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                     std::map<std::string, nvinfer1::Weights> weightMap,
                                     nvinfer1::ITensor& input, std::string lname, float eps);
nvinfer1::ILayer* convBn(nvinfer1::INetworkDefinition* network,
                        std::map<std::string, nvinfer1::Weights> weightMap,
                        nvinfer1::ITensor& input, int ch, int k, int s, 
                        std::string lname, int g = 1);
nvinfer1::IElementWiseLayer* convBnSiLU(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights> weightMap,
                                       nvinfer1::ITensor& input, int ch, 
                                       std::vector<int> k, int s, std::string lname);

// 模块函数
nvinfer1::ILayer* bottleneck(nvinfer1::INetworkDefinition* network,
                            std::map<std::string, nvinfer1::Weights> weightMap,
                            nvinfer1::ITensor& input, int c1, int c2, bool shortcut, 
                            std::vector<int> k1, std::vector<int> k2, float e, std::string lname);
nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights> weightMap,
                                 nvinfer1::ITensor& input, int c1, int c2, int k, std::string lname);
nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network,
                            std::map<std::string, nvinfer1::Weights> weightMap,
                            nvinfer1::ITensor& input, int ch, int grid, 
                            int k, int s, int p, std::string lname);
nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition* network,
                                      std::vector<nvinfer1::IConcatenationLayer*> dets,
                                      const int* strides, int num_outputs,
                                      bool is_segmentation, bool is_pose, bool is_obb);

// YOLOv13 特殊模块
nvinfer1::ILayer* C3k(nvinfer1::INetworkDefinition* network,
                     std::map<std::string, nvinfer1::Weights> weightMap,
                     nvinfer1::ITensor& input, int c1, int c2, int n, 
                     bool shortcut, std::vector<int> k1, std::vector<int> k2, 
                     float e, std::string lname);
nvinfer1::IElementWiseLayer* C3K2(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights>& weightMap,
                                 nvinfer1::ITensor& input, int c1, int c2, int n, 
                                 bool c3k, bool shortcut, float e, std::string lname);
nvinfer1::ILayer* A2C2f(nvinfer1::INetworkDefinition* network,
                       std::map<std::string, nvinfer1::Weights> weightMap,
                       nvinfer1::ITensor& input, int c1, int c2, int n, 
                       bool a2, int area, bool residual, 
                       float mlp_ratio, float e, int g, 
                       bool shortcut, std::string lname);
nvinfer1::ILayer* HyperACE(nvinfer1::INetworkDefinition* network,
                          std::map<std::string, nvinfer1::Weights> weightMap,
                          nvinfer1::ITensor& input, int c1, int c2, std::string lname);
nvinfer1::ILayer* FullPAD_Tunnel(nvinfer1::INetworkDefinition* network,
                                nvinfer1::ITensor& input);
nvinfer1::ILayer* DownsampleConv(nvinfer1::INetworkDefinition* network,
                                std::map<std::string, nvinfer1::Weights> weightMap,
                                nvinfer1::ITensor& input, int c1, int c2, std::string lname);
nvinfer1::ILayer* C3AH(nvinfer1::INetworkDefinition* network,
                      std::map<std::string, nvinfer1::Weights> weightMap,
                      nvinfer1::ITensor& input, int c1, int c2, std::string lname);
nvinfer1::ILayer* ABlock(nvinfer1::INetworkDefinition* network,
                        std::map<std::string, nvinfer1::Weights> weightMap,
                        nvinfer1::ITensor& input, int dim, int num_heads, 
                        float mlp_ratio, int area, std::string lname);
nvinfer1::ILayer* AAttn(nvinfer1::INetworkDefinition* network,
                       std::map<std::string, nvinfer1::Weights> weightMap,
                       nvinfer1::ITensor& input, int dim, int num_heads, 
                       float mlp_ratio, int area, std::string lname);
