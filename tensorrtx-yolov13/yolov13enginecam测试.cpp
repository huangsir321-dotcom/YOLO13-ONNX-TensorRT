#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"
#include <chrono>
#include "config.h"

Logger gLogger;
using namespace nvinfer1;

// ============== 用户自定义配置 ==============
const std::vector<std::string> kClassNames = {
    "cat", 
    "cup", 
    "dog"
    // 在此添加更多类别...
};

const int kCameraIndex = 2;    // 摄像头设备号（0为默认摄像头）
const int kBoxThickness = 2;   // 检测框线条粗细

// ============== 函数声明 ==============
void deserialize_engine(const std::string& engine_name, IRuntime** runtime, 
                       ICudaEngine** engine, IExecutionContext** context);
void prepare_buffers(ICudaEngine* engine, float** input_buffer_device, 
                    float** output_buffer_device, float** output_buffer_host,
                    int& output_size);
void infer_only(IExecutionContext& context, cudaStream_t& stream, void** buffers, 
               float* output_data, int output_size, double& inference_time_ms);
std::vector<Detection> correct_postprocess(float* output, int img_width, int img_height, 
                                          float scale, int pad_w, int pad_h, int num_boxes);
void draw_detections(cv::Mat& img, const std::vector<Detection>& detections, double inference_fps, double total_fps);
cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h, float& scale, int& pad_w, int& pad_h);
std::vector<Detection> nms_cpu(std::vector<Detection>& detections, float iou_threshold);

// ============== 多线程相关结构 ==============
struct FrameData {
    cv::Mat frame;
    cv::Mat preprocessed;
    float scale;
    int pad_w;
    int pad_h;
    std::chrono::time_point<std::chrono::steady_clock> capture_time;
};

std::queue<FrameData> frame_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
std::atomic<bool> stop_processing(false);
const int kQueueSize = 4; // 双缓冲队列大小

// ============== 主函数 ==============
int main() {
    // 打印配置信息
    std::cout << "========== 配置信息 ==========" << std::endl;
    std::cout << "TensorRT 版本: 8.6.1.6" << std::endl;
    std::cout << "摄像头编号: " << kCameraIndex << std::endl;
    std::cout << "检测框线条粗细: " << kBoxThickness << std::endl;
    std::cout << "检测类别 (" << kClassNames.size() << "个):" << std::endl;
    for (size_t i = 0; i < kClassNames.size(); ++i) {
        std::cout << "  " << i << ": " << kClassNames[i] << std::endl;
    }
    std::cout << "=============================" << std::endl;

    // 初始化CUDA - 添加详细错误检查
    cudaError_t cuda_status = cudaSetDevice(kGpuId);
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA设置设备失败: " << cudaGetErrorString(cuda_status) << std::endl;
        std::cerr << "请检查: " << std::endl;
        std::cerr << "1. GPU设备是否存在 (运行 nvidia-smi)" << std::endl;
        std::cerr << "2. 设备号kGpuId是否正确 (当前值: " << kGpuId << ")" << std::endl;
        std::cerr << "3. CUDA驱动是否正确安装" << std::endl;
        return -1;
    }
    
    // 测试CUDA是否可用 - 修复类型转换问题
    void* test_ptr = nullptr;
    cuda_status = cudaMalloc(&test_ptr, sizeof(float));
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA测试分配失败: " << cudaGetErrorString(cuda_status) << std::endl;
        return -1;
    }
    cudaFree(test_ptr);
    std::cout << "CUDA初始化成功，使用GPU设备: " << kGpuId << std::endl;

    // 加载TensorRT引擎
    std::string engine_name = "/home/bpz/yolo系列网课/tensorrtx-yolov13/yolov13.engine";
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);

    // 准备缓冲区
    float* input_buffer_device = nullptr;
    float* output_buffer_device = nullptr;
    float* output_buffer_host = nullptr;
    int output_size = 0;
    prepare_buffers(engine, &input_buffer_device, &output_buffer_device, &output_buffer_host, output_size);

    // 动态计算候选框数量
    const int num_classes = kClassNames.size();
    const int elements_per_box = 4 + num_classes;
    
    if (output_size % elements_per_box != 0) {
        std::cerr << "错误: 输出大小(" << output_size << ")不是每个框元素数(" 
                  << elements_per_box << ")的整数倍!" << std::endl;
        return -1;
    }
    
    const int num_boxes = output_size / elements_per_box;
    std::cout << "动态计算候选框数量: " << num_boxes << " (输出大小: " 
              << output_size << ", 每框元素: " << elements_per_box << ")" << std::endl;

    // 创建CUDA流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kInputH * kInputW * 3);

    // 初始化摄像头
    cv::VideoCapture cap;
    if(!cap.open(kCameraIndex, cv::CAP_V4L2)) {
        std::cerr << "ERROR: 无法打开摄像头 " << kCameraIndex << std::endl;
        std::cerr << "请尝试以下解决方案: " << std::endl;
        std::cerr << "1. 检查摄像头是否连接正常" << std::endl;
        std::cerr << "2. 检查设备号是否正确 (尝试 0, 1, 2...)" << std::endl;
        std::cerr << "3. 确保用户有访问摄像头的权限 (尝试: sudo chmod a+rw /dev/video*)" << std::endl;
        return -1;
    }

    // 设置摄像头分辨率
    int frame_width = 640;
    int frame_height = 480;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);
    cap.set(cv::CAP_PROP_FPS, 60); // 提高摄像头帧率

    // 验证摄像头设置
    double actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "摄像头实际设置: " << actual_width << "x" << actual_height 
              << " @ " << actual_fps << " FPS" << std::endl;

    cv::namedWindow("YOLOv13 Detection", cv::WINDOW_NORMAL);

    // 帧率统计变量
    double total_time = 0;
    int frame_count = 0;
    double avg_fps = 0;
    double avg_inference_time = 0;

    // 启动预处理线程
    std::thread preprocessing_thread([&]() {
        // 设置当前线程的CUDA设备
        cudaSetDevice(kGpuId);
        
        while (!stop_processing) {
            cv::Mat frame;
            if (!cap.read(frame)) {
                std::cerr << "ERROR: 无法读取帧" << std::endl;
                break;
            }

            if (frame.empty()) {
                std::cerr << "警告: 获取到空帧，跳过处理" << std::endl;
                continue;
            }

            auto capture_time = std::chrono::steady_clock::now();
            
            // 预处理 - 使用letterbox保持宽高比
            float scale = 0.0f;
            int pad_w = 0;
            int pad_h = 0;
            cv::Mat preprocessed = preprocess_img(frame, kInputW, kInputH, scale, pad_w, pad_h);
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, []{ return frame_queue.size() < kQueueSize; });
                
                frame_queue.push({frame.clone(), preprocessed.clone(), scale, pad_w, pad_h, capture_time});
            }
            queue_cv.notify_one();
        }
    });

    // 主处理循环
    while (true) {
        FrameData frame_data;
        bool frame_available = false;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (queue_cv.wait_for(lock, std::chrono::milliseconds(100), 
                [&]{ return !frame_queue.empty(); })) {
                
                frame_data = frame_queue.front();
                frame_queue.pop();
                frame_available = true;
            }
        }
        
        if (!frame_available) {
            continue; // 跳过空帧
        }
        
        queue_cv.notify_one();

        auto total_start = std::chrono::steady_clock::now();
        
        // 将预处理后的图像复制到GPU - 异步执行
        std::vector<cv::Mat> img_batch = {frame_data.preprocessed};
        cuda_batch_preprocess(img_batch, input_buffer_device, kInputW, kInputH, stream);

        // 推理
        double inference_time_ms = 0;
        void* buffers[2] = {input_buffer_device, output_buffer_device};
        infer_only(*context, stream, buffers, output_buffer_host, output_size, inference_time_ms);

        // 后处理 (传递动态计算的num_boxes)
        auto postprocess_start = std::chrono::steady_clock::now();
        std::vector<Detection> detections = correct_postprocess(output_buffer_host, 
                                                              frame_data.frame.cols, 
                                                              frame_data.frame.rows, 
                                                              frame_data.scale, 
                                                              frame_data.pad_w, 
                                                              frame_data.pad_h, 
                                                              num_boxes);
        auto postprocess_end = std::chrono::steady_clock::now();
        double postprocess_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            postprocess_end - postprocess_start).count() / 1000.0;

        // 绘制结果
        auto draw_start = std::chrono::steady_clock::now();
        auto total_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            total_end - frame_data.capture_time).count() / 1000.0;
        double total_fps = 1000.0 / (total_ms + 1e-6);
        double inference_fps = 1000.0 / (inference_time_ms + 1e-6);
        
        // 计算平均帧率
        total_time += total_ms;
        avg_inference_time += inference_time_ms;
        frame_count++;
        
        if (frame_count >= 30) {
            avg_fps = 1000.0 / (total_time / frame_count);
            avg_inference_time = avg_inference_time / frame_count;
            frame_count = 0;
            total_time = 0;
            avg_inference_time = 0;
        }
        
        draw_detections(frame_data.frame, detections, inference_fps, avg_fps);
        auto draw_end = std::chrono::steady_clock::now();
        double draw_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            draw_end - draw_start).count() / 1000.0;

        // 显示
        cv::imshow("YOLOv13 Detection", frame_data.frame);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 'q' 或 ESC 键
            stop_processing = true;
            break;
        }
    }

    // 清理资源
    stop_processing = true;
    preprocessing_thread.join();
    cap.release();
    cv::destroyAllWindows();
    CUDA_CHECK(cudaFree(input_buffer_device));
    CUDA_CHECK(cudaFree(output_buffer_device));
    delete[] output_buffer_host;
    cuda_preprocess_destroy();
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete context;
    delete engine;
    delete runtime;

    return 0;
}

// ============== 函数实现 ==============
void deserialize_engine(const std::string& engine_name, IRuntime** runtime, 
                       ICudaEngine** engine, IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to read engine file: " << engine_name << std::endl;
        std::cerr << "请检查文件路径是否正确，文件是否存在" << std::endl;
        exit(1);
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    if (!*runtime) {
        std::cerr << "Failed to create InferRuntime" << std::endl;
        exit(1);
    }
    
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    if (!*engine) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        std::cerr << "可能原因: " << std::endl;
        std::cerr << "1. 引擎文件与当前TensorRT版本不兼容" << std::endl;
        std::cerr << "2. 引擎文件损坏" << std::endl;
        exit(1);
    }
    
    *context = (*engine)->createExecutionContext();
    if (!*context) {
        std::cerr << "Failed to create execution context" << std::endl;
        exit(1);
    }
    
    delete[] serialized_engine;
}

void prepare_buffers(ICudaEngine* engine, float** input_buffer_device, 
                    float** output_buffer_device, float** output_buffer_host,
                    int& output_size) {
    const int num_bindings = engine->getNbBindings();
    
    if (num_bindings != 2) {
        std::cerr << "Error: Expected 2 bindings, got " << num_bindings << std::endl;
        exit(1);
    }
    
    // 分配输入缓冲区
    Dims input_dims = engine->getBindingDimensions(0);
    int input_size = 1;
    for (int j = 0; j < input_dims.nbDims; j++) {
        input_size *= input_dims.d[j];
    }
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, input_size * sizeof(float)));
    
    // 分配输出缓冲区
    Dims output_dims = engine->getBindingDimensions(1);
    output_size = 1;
    for (int j = 0; j < output_dims.nbDims; j++) {
        output_size *= output_dims.d[j];
    }
    
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, output_size * sizeof(float)));
    *output_buffer_host = new float[output_size];
    
    // 输出维度信息用于调试
    std::cout << "输入维度: [";
    for (int i = 0; i < input_dims.nbDims; ++i) {
        std::cout << input_dims.d[i];
        if (i < input_dims.nbDims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "输出维度: [";
    for (int i = 0; i < output_dims.nbDims; ++i) {
        std::cout << output_dims.d[i];
        if (i < output_dims.nbDims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void infer_only(IExecutionContext& context, cudaStream_t& stream, void** buffers, 
               float* output_data, int output_size, double& inference_time_ms) {
    auto start = std::chrono::steady_clock::now();
    
    // 异步执行推理
    if (!context.enqueueV2(buffers, stream, nullptr)) {
        std::cerr << "ERROR: 推理执行失败" << std::endl;
        return;
    }
    
    // 异步拷贝输出数据
    CUDA_CHECK(cudaMemcpyAsync(output_data, buffers[1], output_size * sizeof(float), 
               cudaMemcpyDeviceToHost, stream));
    
    // 等待流完成
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::steady_clock::now();
    inference_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h, float& scale, int& pad_w, int& pad_h) {
    // 计算缩放比例 (保持宽高比)
    int img_w = img.cols;
    int img_h = img.rows;
    scale = std::min(static_cast<float>(input_w) / img_w, 
                    static_cast<float>(input_h) / img_h);
    
    // 计算新尺寸
    int new_w = static_cast<int>(img_w * scale);
    int new_h = static_cast<int>(img_h * scale);
    
    // 计算填充
    pad_w = (input_w - new_w) / 2;
    pad_h = (input_h - new_h) / 2;
    
    // 缩放图像
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    
    // 创建填充后的图像
    cv::Mat out = cv::Mat::zeros(input_h, input_w, CV_8UC3);
    out.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_w, pad_h, new_w, new_h)));
    
    return out;
}

// 优化后的后处理函数 - 修复窄化转换问题
std::vector<Detection> correct_postprocess(float* output, int img_width, int img_height, 
                                          float scale, int pad_w, int pad_h, int num_boxes) {
    std::vector<Detection> detections;
    detections.reserve(num_boxes); // 预分配内存
    
    // 获取类别数量
    const int num_classes = kClassNames.size();
    const int elements_per_box = 4 + num_classes;
    
    // 提前计算逆缩放因子
    const float inv_scale = 1.0f / scale;
    
    for (int i = 0; i < num_boxes; ++i) {
        // 直接访问数组元素，避免多次指针计算
        const float* box_ptr = output + i;
        const float x_c = box_ptr[0];
        const float y_c = box_ptr[num_boxes];
        const float w = box_ptr[2 * num_boxes];
        const float h = box_ptr[3 * num_boxes];
        
        // 找到最高置信度的类别
        float max_conf = 0;
        int class_id = -1;
        
        // 使用指针直接访问类别分数部分
        const float* scores_ptr = output + 4 * num_boxes;
        for (int c = 0; c < num_classes; ++c) {
            const float cls_conf = scores_ptr[c * num_boxes + i];
            if (cls_conf > max_conf) {
                max_conf = cls_conf;
                class_id = c;
            }
        }
        
        // 提前过滤低置信度的检测
        if (max_conf < kConfThresh || class_id == -1) continue;
        
        // 坐标转换: 从letterbox坐标转回原始图像坐标
        const float x_center = (x_c - pad_w) * inv_scale;
        const float y_center = (y_c - pad_h) * inv_scale;
        const float abs_w = w * inv_scale;
        const float abs_h = h * inv_scale;
        
        // 计算边界框坐标
        float x1 = x_center - abs_w / 2;
        float y1 = y_center - abs_h / 2;
        float x2 = x_center + abs_w / 2;
        float y2 = y_center + abs_h / 2;
        
        // 边界检查
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_width - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_height - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_width - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_height - 1)));
        
        // 检查边界框有效性
        if (x1 >= x2 || y1 >= y2) continue;
        
        // 修复窄化转换问题：将class_id显式转换为float
        Detection det = {
            {x1, y1, x2, y2},
            max_conf,
            static_cast<float>(class_id)  // 显式类型转换
        };
        detections.push_back(det);
    }
    
    // 应用NMS
    if (!detections.empty()) {
        return nms_cpu(detections, kNmsThresh);
    }
    return detections;
}

// 优化后的NMS实现
std::vector<Detection> nms_cpu(std::vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) return {};
    
    // 按置信度降序排序
    std::sort(detections.begin(), detections.end(), 
             [](const Detection& a, const Detection& b) { 
                 return a.conf > b.conf; 
             });
    
    // 预计算所有区域
    std::vector<float> areas;
    areas.reserve(detections.size());
    for (const auto& det : detections) {
        const float w = det.bbox[2] - det.bbox[0];
        const float h = det.bbox[3] - det.bbox[1];
        areas.push_back(w * h);
    }
    
    std::vector<bool> keep(detections.size(), true);
    std::vector<Detection> final_detections;
    final_detections.reserve(detections.size() / 2); // 预分配内存
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!keep[i]) continue;
        
        final_detections.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!keep[j] || detections[i].class_id != detections[j].class_id) 
                continue;
            
            const float* bbox_i = detections[i].bbox;
            const float* bbox_j = detections[j].bbox;
            
            // 计算交集
            const float inter_x1 = std::max(bbox_i[0], bbox_j[0]);
            const float inter_y1 = std::max(bbox_i[1], bbox_j[1]);
            const float inter_x2 = std::min(bbox_i[2], bbox_j[2]);
            const float inter_y2 = std::min(bbox_i[3], bbox_j[3]);
            
            const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
            const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
            const float inter_area = inter_w * inter_h;
            
            const float iou = inter_area / (areas[i] + areas[j] - inter_area + 1e-6f);
            
            if (iou > iou_threshold) {
                keep[j] = false;
            }
        }
    }
    
    return final_detections;
}

// 优化后的绘制函数
void draw_detections(cv::Mat& img, const std::vector<Detection>& detections, double inference_fps, double avg_fps) {
    // 显示帧率信息
    std::string fps_info = cv::format("FPS: %.1f (%.1f avg)", inference_fps, avg_fps);
    cv::putText(img, fps_info, cv::Point(20, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    // 显示摄像头编号
    std::string cam_info = cv::format("Cam %d", kCameraIndex);
    cv::putText(img, cam_info, cv::Point(20, 60), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    // 为每个类别定义不同颜色
    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),   // 红色
        cv::Scalar(0, 255, 0),   // 绿色
        cv::Scalar(255, 0, 0),   // 蓝色
        cv::Scalar(0, 255, 255), // 黄色
        cv::Scalar(255, 0, 255), // 紫色
        cv::Scalar(255, 255, 0)  // 青色
    };

    for (const auto& det : detections) {
        // 注意：Detection结构体中的class_id是float类型
        int class_id = static_cast<int>(det.class_id);
        if (class_id < 0 || class_id >= static_cast<int>(kClassNames.size())) continue;

        int x1 = static_cast<int>(det.bbox[0]);
        int y1 = static_cast<int>(det.bbox[1]);
        int x2 = static_cast<int>(det.bbox[2]);
        int y2 = static_cast<int>(det.bbox[3]);

        // 为每个类别分配颜色
        cv::Scalar color = colors[class_id % colors.size()];
        
        // 绘制边界框
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, kBoxThickness);
        
        // 创建标签文本
        std::string label = cv::format("%s %.2f", kClassNames[class_id].c_str(), det.conf);
        
        // 计算文本大小和背景位置
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Rect text_bg(x1, y1 - text_size.height - 5, text_size.width, text_size.height + 5);
        
        // 绘制文本背景
        cv::rectangle(img, text_bg, color, cv::FILLED);
        
        // 绘制文本
        cv::putText(img, label, cv::Point(x1, y1 - 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}
