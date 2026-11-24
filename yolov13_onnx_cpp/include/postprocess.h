#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// 类别定义（顺序必须与训练时一致）
static const std::vector<std::string> CLASS_NAMES = {"cat", "cup", "dog"};

struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
    std::string class_name;
};

std::vector<Detection> postprocess(
    const std::vector<float>& output_data,
    const std::vector<int64_t>& output_shape,
    const cv::Size& original_image_size,
    float conf_threshold = 0.3,
    float iou_threshold = 0.3
);
