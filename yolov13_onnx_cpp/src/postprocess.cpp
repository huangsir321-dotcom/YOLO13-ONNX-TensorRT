#include "postprocess.h"
#include <algorithm>

static float iou(const cv::Rect& a, const cv::Rect& b) {
    float inter = (a & b).area();
    return inter / (a.area() + b.area() - inter);
}

std::vector<Detection> postprocess(
    const std::vector<float>& output_data,
    const std::vector<int64_t>& output_shape,
    const cv::Size& img_size,
    float conf_thresh, float iou_thresh) {
    
    std::vector<Detection> detections;
    const int num_classes = CLASS_NAMES.size();
    const int num_boxes = output_shape[2];
    
    // 解析输出数据 [1,7,N] (7=4坐标+3类别分数)
    for (int i = 0; i < num_boxes; ++i) {
        // 获取坐标
        float x_center = output_data[i];
        float y_center = output_data[num_boxes + i];
        float width = output_data[2 * num_boxes + i];
        float height = output_data[3 * num_boxes + i];
        
        // 坐标转换
        float scale = std::min(640.0f / img_size.width, 640.0f / img_size.height);
        x_center = (x_center - (640 - img_size.width * scale) / 2) / scale;
        y_center = (y_center - (640 - img_size.height * scale) / 2) / scale;
        width /= scale;
        height /= scale;
        
        // 计算边界框
        float x1 = std::max(0.0f, x_center - width / 2);
        float y1 = std::max(0.0f, y_center - height / 2);
        float x2 = std::min(x_center + width / 2, (float)img_size.width);
        float y2 = std::min(y_center + height / 2, (float)img_size.height);
        
        // 获取类别
        int class_id = -1;
        float max_score = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = output_data[(4 + c) * num_boxes + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        
        if (max_score > conf_thresh) {
            detections.push_back({
                cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)),
                max_score,
                class_id,
                CLASS_NAMES[class_id]  // 填充类别名称
            });
        }
    }
    
    // 非极大抑制
    std::vector<Detection> results;
    std::sort(detections.begin(), detections.end(), 
        [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    
    for (const auto& det : detections) {
        bool keep = true;
        for (const auto& res : results) {
            if (det.class_id == res.class_id && iou(det.box, res.box) > iou_thresh) {
                keep = false;
                break;
            }
        }
        if (keep) results.push_back(det);
    }
    
    return results;
}
