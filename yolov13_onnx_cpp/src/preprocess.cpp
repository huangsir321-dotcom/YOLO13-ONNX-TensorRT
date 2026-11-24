#include "preprocess.h"
#include <opencv2/opencv.hpp>

cv::Mat preprocess_image(const cv::Mat& image, const std::vector<int64_t>& input_shape) {
    // 输入形状应为 [batch, channel, height, width]
    int input_height = input_shape[2];
    int input_width = input_shape[3];
    
    // 调整大小并保持宽高比
    float scale = std::min(input_width / (float)image.cols, input_height / (float)image.rows);
    int new_width = int(image.cols * scale);
    int new_height = int(image.rows * scale);
    
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height));
    
    // 填充到模型输入尺寸
    int dw = input_width - new_width;
    int dh = input_height - new_height;
    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;
    
    cv::Mat padded_image;
    cv::copyMakeBorder(resized_image, padded_image, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    return padded_image;
}

std::vector<float> prepare_input_tensor(const cv::Mat& resized_image) {
    // 将图像转换为CHW格式并归一化
    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
    
    // 分离通道
    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    
    // 合并为连续的一维数组 (CHW格式)
    std::vector<float> input_tensor;
    for (const auto& channel : channels) {
        input_tensor.insert(input_tensor.end(), 
                           channel.ptr<float>(), 
                           channel.ptr<float>() + channel.total());
    }
    
    return input_tensor;
}
