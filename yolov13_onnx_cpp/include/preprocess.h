#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat preprocess_image(const cv::Mat& image, const std::vector<int64_t>& input_shape);
std::vector<float> prepare_input_tensor(const cv::Mat& resized_image);
