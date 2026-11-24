#include <iostream>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include "preprocess.h"
#include "postprocess.h"
#include "utils.h"
#include <chrono>  // 用于帧率计算
#include <omp.h>    // 用于多线程

int main() {
    // 需要释放的资源
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    Ort::Env env(nullptr); // 延迟初始化
    
    try {
        // 1. 初始化ONNX Runtime环境
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX-Inference");

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

        // 4. 打开摄像头（默认设备索引为 0，外接摄像头通常是 1 或 2）
        cv::VideoCapture cap(2);  // 修改为 1 或 2 如果你使用外接摄像头
        if (!cap.isOpened()) {
            std::cerr << "无法打开摄像头！" << std::endl;
            return -1;
        }

        // 设置摄像头分辨率（可选）
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        std::cout << "开始读取摄像头画面，按任意键退出..." << std::endl;

        // 记录帧率用变量
        int frame_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "无法读取摄像头画面！" << std::endl;
                break;
            }

            // 记录开始时间
            auto inference_start_time = std::chrono::high_resolution_clock::now();

            // 获取原始图像尺寸
            cv::Size original_size = frame.size();

            // 5. 预处理
            cv::Mat resized_image;
#pragma omp parallel sections
            {
#pragma omp section
                resized_image = preprocess_image(frame, input_shape);
            }

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
            std::vector<Detection> detections;
#pragma omp parallel sections
            {
#pragma omp section
                detections = postprocess(output_vector, output_shape, original_size, 0.5, 0.5);
            }

            // 10. 绘制结果
            for (const auto& det : detections) {
                // 绘制边界框（绿色）
                cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

                // 构建标签文本 "类别:置信度"
                std::string label = det.class_name + ":" + cv::format("%.2f", det.conf);

                // 计算文字背景框大小
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label,
                                                    cv::FONT_HERSHEY_SIMPLEX,
                                                    0.5, 1, &baseline);

                // 绘制文字背景（黑色半透明）
                cv::rectangle(frame,
                             cv::Point(det.box.x, det.box.y - text_size.height - 5),
                             cv::Point(det.box.x + text_size.width, det.box.y),
                             cv::Scalar(0, 0, 0), cv::FILLED);

                // 绘制文字（白色）
                cv::putText(frame, label,
                           cv::Point(det.box.x, det.box.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX,
                           0.5, cv::Scalar(255, 255, 255), 1);
            }

            // 显示帧率
            frame_count++;
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            if (elapsed_time >= 1) { // 每秒更新一次帧率
                double fps = frame_count / static_cast<double>(elapsed_time);
                std::ostringstream ss;
                ss << "FPS: " << fps;
                std::string fps_text = ss.str();

                // 在视频帧的右上角绘制帧率
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::rectangle(frame, 
                              cv::Point(frame.cols - text_size.width - 10, 10),
                              cv::Point(frame.cols - 10, 10 + text_size.height + 5),
                              cv::Scalar(0, 0, 0), cv::FILLED);
                cv::putText(frame, fps_text, 
                          cv::Point(frame.cols - text_size.width - 10, 10 + text_size.height + baseline - 5),
                          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                frame_count = 0; // 重置计数器
                start_time = current_time; // 更新起始时间
            }

            // 11. 显示结果
            cv::imshow("Detection Results", frame);

            // 按任意键退出
            if (cv::waitKey(1) >= 0) {
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();

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


