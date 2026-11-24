#include <iostream>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include "preprocess.h"
#include "postprocess.h"
#include "utils.h"

int main() {
    // 需要释放的资源
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    Ort::Env env(nullptr); // 延迟初始化
    
    try {
        // 1. 初始化ONNX Runtime环境
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv13-ONNX-Inference");

        // 2. 加载模型
        const char* model_path = "/home/bpz/yolo系列网课/yolov13_onnx_cpp/models/best.onnx";
        Ort::Session session = create_session(env, model_path);

        // 3. 获取模型信息
        input_names = get_input_names(session);
        output_names = get_output_names(session);
        auto input_shape = get_input_shape(session);
        auto output_shape = get_output_shape(session);

        // 打印调试信息
        std::cout << "Input shape: ";
        for (auto dim : input_shape) std::cout << dim << " ";
        std::cout << "\nOutput shape: ";
        for (auto dim : output_shape) std::cout << dim << " ";
        std::cout << std::endl;

        // 4. 加载测试图像
        std::string image_path = "/home/bpz/yolo系列网课/yolov13_onnx_cpp/test_images/test.jpg";
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return -1;
        }
        cv::Size original_size = image.size();

        // 5. 预处理
        cv::Mat resized_image = preprocess_image(image, input_shape);
        std::vector<float> input_tensor = prepare_input_tensor(resized_image);

        // 6. 创建输入Tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, input_tensor.data(), input_tensor.size(),
            input_shape.data(), input_shape.size()));

        // 7. 运行推理
        std::vector<Ort::Value> output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_names.size(),
            output_names.data(), output_names.size());

        // 8. 获取输出数据
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> output_vector(output_data, output_data + output_size);

        // 9. 后处理
        std::vector<Detection> detections = postprocess(
            output_vector, output_shape, original_size, 0.5, 0.5);

        // 10. 绘制结果
        for (const auto& det : detections) {
            // 绘制边界框（绿色）
            cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
            
            // 构建标签文本 "类别:置信度"
            std::string label = det.class_name + ":" + cv::format("%.2f", det.conf);
            
            // 计算文字背景框大小
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, 
                                               cv::FONT_HERSHEY_SIMPLEX,
                                               0.5, 1, &baseline);
            
            // 绘制文字背景（黑色半透明）
            cv::rectangle(image, 
                         cv::Point(det.box.x, det.box.y - text_size.height - 5),
                         cv::Point(det.box.x + text_size.width, det.box.y),
                         cv::Scalar(0, 0, 0), cv::FILLED);
            
            // 绘制文字（白色）
            cv::putText(image, label,
                       cv::Point(det.box.x, det.box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX,
                       0.5, cv::Scalar(255, 255, 255), 1);
        }

        // 11. 显示和保存结果
        cv::imshow("Detection Results", image);
        cv::waitKey(0);
        cv::imwrite("detection_result.jpg", image);

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    // 12. 释放资源
    Ort::AllocatorWithDefaultOptions allocator;
    for (auto name : input_names) {
        allocator.Free(const_cast<void*>(static_cast<const void*>(name)));
    }
    for (auto name : output_names) {
        allocator.Free(const_cast<void*>(static_cast<const void*>(name)));
    }

    return 0;
}
