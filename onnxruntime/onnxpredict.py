# predict.py
import cv2
import numpy as np
import onnxruntime as ort
import argparse

# 类别标签（按你的数据集顺序）
CLASSES = ['cup', 'dog', 'cat']

# 颜色调色板
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def preprocess(image, input_size=(640, 640)):
    """图像预处理"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img / 255.0  # 归一化
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = img[np.newaxis, ...].astype(np.float32)  # 添加 batch 维度
    return img

def nms(boxes, scores, threshold=0.5):
    """非极大值抑制"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep

def draw_detections(image, boxes, scores, class_ids, conf_thres):
    """绘制检测框和标签"""
    for i in range(len(boxes)):
        if scores[i] < conf_thres:
            continue
        box = boxes[i]
        class_id = class_ids[i]
        x, y, w, h = box
        color = COLORS[class_id % len(COLORS)]
        label = f"{CLASSES[class_id]}: {scores[i]:.2f}"

        # 绘制边界框
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

        # 绘制标签
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (int(x), int(y) - th), (int(x) + tw, int(y)), color, -1)
        cv2.putText(image, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main(args):
    # 加载 ONNX 模型
    session = ort.InferenceSession(args.model)

    # 获取输入输出名称
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    # 读取图像
    image = cv2.imread(args.image)
    orig_h, orig_w = image.shape[:2]

    # 预处理
    input_data = preprocess(image)
    outputs = session.run(output_names, {input_name: input_data})

    # 后处理
    output = outputs[0].squeeze()  # 假设输出为 (1, 8400, 84) -> (8400, 84)
    boxes = output[:, :4]
    scores = np.max(output[:, 4:], axis=1)
    class_ids = np.argmax(output[:, 4:], axis=1)

    # 过滤低置信度
    valid_indices = scores > args.conf_thres
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    class_ids = class_ids[valid_indices]

    # NMS
    indices = nms(boxes, scores, threshold=args.iou_thres)

    # 缩放回原始图像尺寸
    boxes[:, [0, 2]] *= orig_w / 640
    boxes[:, [1, 3]] *= orig_h / 640

    # 绘制检测结果
    draw_detections(image, boxes[indices], scores[indices], class_ids[indices], args.conf_thres)

    # 显示并保存结果
    cv2.imshow("YOLOv8 ONNX Inference", image)
    cv2.imwrite("output.jpg", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="/home/bpz/yolo系列网课/test_cup.jpg", help="输入图像路径")
    parser.add_argument("--model", type=str, default="/home/bpz/yolo系列网课/onnxruntime/yolov13n.onnx", help="ONNX 模型路径")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU 阈值")
    args = parser.parse_args()

    main(args)
