
import os
import cv2
import numpy as np
import onnxruntime as ort


class OnnxRun:
    def __init__(self, onnx_model):
        self.model = onnx_model  # 模型
        self.conf_thres = 0.1  # 置信度
        self.iou_thres = 0.1  # iou置信度
        # 缺陷类别信息
        self.classes = {0: 'cat', 1: 'cup', 2: 'dog'}
        # 类别颜色
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # 推理会话
        #self.session = ort.InferenceSession(self.model, providers=['CPUExecutionProvider'])

        providers = ort.get_available_providers()
        print("可用的 ONNX Runtime 提供者:", providers)
        # 优先使用 CUDA，否则使用 CPU
        provider_to_use = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in providers else ['CPUExecutionProvider']

        self.session = ort.InferenceSession(self.model, providers=provider_to_use)


        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        # 获取模型输入图像的大小
        input_shape = self.session.get_inputs()[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        # 预热
        image = np.zeros((1, 3, self.input_width, self.input_height), dtype=np.float32)
        self.session.run(None, {self.input_name: image})
        print("INIT FINISHED")

    '''图像预处理'''
    def preprocess(self, image_path):
        # 读取输入图像
        self.img = cv2.imread(image_path)
        # 获取图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]
        # 转换图像颜色空间
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # 调整图像大小
        img = cv2.resize(img, (self.input_width, self.input_height))
        # 归一化图像数据
        image_data = np.array(img) / 255.0
        # 调整图像数据维度顺序
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        # 扩展图像数据维度
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        # 返回预处理后的图像数据
        return image_data

    '''模型推理'''
    def onnx_run(self, imgs_dir):
        self.det_mes = []
        save_path = 'results'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for img in os.listdir(imgs_dir):
            img_path = os.path.join(imgs_dir, img)
            img_data = self.preprocess(img_path)
            outputs = self.session.run(None, {self.input_name: img_data})
            img_result = self.postprocess(outputs)  # Removed the extra argument
            # 保存
            cv2.imwrite(f"{save_path}/{img}", img_result)

    def postprocess(self, output):  # Removed the extra parameter
        # 转换输出形状
        outputs = np.squeeze(output[0]).T
        # 计算缩放因子
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height
        # 提取所有检测结果的类别分数
        classes_scores = outputs[:, 4:]
        # 找到每个检测结果的最大分数
        max_scores = np.amax(classes_scores, axis=1)
        # 找到最大分数对应的类别ID
        class_ids = np.argmax(classes_scores, axis=1)
        # 过滤掉低于置信度阈值的检测结果
        valid_indices = np.where(max_scores >= self.conf_thres)[0]
        valid_outputs = outputs[valid_indices]
        valid_class_ids = class_ids[valid_indices]
        valid_scores = max_scores[valid_indices]
        # 提取有效的检测框坐标
        x = valid_outputs[:, 0]
        y = valid_outputs[:, 1]
        w = valid_outputs[:, 2]
        h = valid_outputs[:, 3]
        # 计算缩放后的检测框坐标
        left = (x - w / 2) * x_factor
        top = (y - h / 2) * y_factor
        width = w * x_factor
        height = h * y_factor
        # 将坐标转换为整数
        boxes = np.column_stack((left, top, width, height)).astype(int)
        # 应用非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, valid_scores, self.conf_thres, self.iou_thres)
        # 绘制检测框和计算检测结果
        if len(indices) != 0:
            for i in indices:
                box = boxes[i]
                score = valid_scores[i]
                class_id = valid_class_ids[i]
                self.draw_detections(self.img, box, score, class_id)

        return self.img

    '''绘制检测框'''
    def draw_detections(self, img, box, score, class_id):
        # 获取边框信息
        x1, y1, w, h = box
        # 获取颜色
        color = self.color_palette[class_id]
        # 框的位置、颜色、粗细
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        # 标签
        # label = f"{self.classes[class_id]}: {score:.2f}"
        label = f"{self.classes[class_id]}"
        # 标签设置：label标签文本、cv2.FONT_HERSHEY_SIMPLEX字体类型、0.5字体缩放比例、1字体粗细
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        # 绘制标签背景矩形：(label_x, label_y - label_height)背景矩形的左上角坐标、(label_x + label_width, label_y + label_height)背景矩形的右下角坐标
        # color背景矩形的颜色、cv2.FILLED背景矩形填充
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )
        # 绘制标签文本：img输入图像、label标签文本、(label_x, label_y)标签文本左上角、cv2.FONT_HERSHEY_  SIMPLEX字体类型、
        # 0.5字体缩放比例、(255, 255, 255)文本颜色、1字体粗细、cv2.LINE_AA抗锯齿线型
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


onnx = OnnxRun("/home/bpz/yolo系列网课/yolov13-main/runs/train/exp3/weights/best.onnx")
onnx.onnx_run("/home/bpz/yolo系列网课/images/img/")