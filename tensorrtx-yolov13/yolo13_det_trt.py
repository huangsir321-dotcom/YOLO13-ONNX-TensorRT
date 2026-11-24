import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple, Optional

# 常量配置
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CLASS_NAMES = ["cat", "cup", "dog"]  # nc=3

class YOLOv13TRT:
    def __init__(self, engine_path: str):
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()

        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine.")
        
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # 使用新 API 获取绑定信息
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)

            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'host': host_mem, 'device': device_mem})
                self.input_shape = tuple(shape)
            else:
                self.outputs.append({'name': name, 'host': host_mem, 'device': device_mem})

        print(f"Engine initialized. Input shape: {self.input_shape}")
        print(f"Output tensors: {[out['name'] for out in self.outputs]}")

    def infer(self, image: np.ndarray) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            print("Error: Input image is None or empty.")
            return None

        try:
            self.ctx.push()

            # 1. 预处理
            input_image, image_raw, orig_h, orig_w = self._preprocess(image)

            # 2. 推理
            np.copyto(self.inputs[0]['host'], input_image.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

            # 设置输入输出绑定
            for inp in self.inputs:
                self.context.set_tensor_address(inp['name'], inp['device'])
            for out in self.outputs:
                self.context.set_tensor_address(out['name'], out['device'])

            # 执行推理（使用新 API）
            self.context.execute_async_v3(self.stream.handle)

            # 拷贝输出
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()

            # 3. 后处理
            output_data = self.outputs[0]['host']
            print(f"Raw output shape: {output_data.shape}")  # 应该是 (5880,) -> (7,8400)

            # 重新 reshape 为 (7, 8400)
            output_data = output_data.reshape(7, -1)  # (7, 8400)
            print(f"Reshaped output: {output_data.shape}")

            detections = self._postprocess(output_data, orig_h, orig_w)

            # 4. 绘图
            result_image = self._draw_boxes(image_raw, detections)

            return result_image

        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.ctx.pop()

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        orig_h, orig_w = image.shape[:2]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        r = min(INPUT_WIDTH / orig_w, INPUT_HEIGHT / orig_h)
        new_w = int(orig_w * r)
        new_h = int(orig_h * r)
        pad_w = (INPUT_WIDTH - new_w) // 2
        pad_h = (INPUT_HEIGHT - new_h) // 2

        img = cv2.resize(img, (new_w, new_h))
        img = cv2.copyMakeBorder(
            img, pad_h, pad_h, pad_w, pad_w, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return np.ascontiguousarray(img), image, orig_h, orig_w

    def _postprocess(self, output: np.ndarray, orig_h: int, orig_w: int) -> List[List[float]]:
        """正确处理 (7, 8400) 输出"""
        try:
            if output.size == 0:
                return []

            # output: (7, 8400) -> 4 + 3
            num_boxes = output.shape[1]  # 8400
            x_c = output[0, :]  # 中心 x
            y_c = output[1, :]  # 中心 y
            w = output[2, :]    # 宽
            h = output[3, :]    # 高
            conf = output[4:7, :]  # (3, 8400), nc=3

            # 计算最大置信度和类别 ID
            max_conf = np.max(conf, axis=0)  # (8400,)
            cls_id = np.argmax(conf, axis=0)  # (8400,) -> 0,1,2

            # 过滤低置信度
            keep = max_conf > CONF_THRESH
            if not np.any(keep):
                return []

            # 坐标转换
            r = min(INPUT_WIDTH / orig_w, INPUT_HEIGHT / orig_h)
            pad_w = (INPUT_WIDTH - r * orig_w) / 2
            pad_h = (INPUT_HEIGHT - r * orig_h) / 2

            valid_boxes = []
            for i in range(num_boxes):
                if keep[i]:
                    # 转为左上右下
                    x1 = x_c[i] - w[i] / 2
                    y1 = y_c[i] - h[i] / 2
                    x2 = x_c[i] + w[i] / 2
                    y2 = y_c[i] + h[i] / 2

                    # 去填充并缩放回原图
                    x1 = (x1 - pad_w) / r
                    y1 = (y1 - pad_h) / r
                    x2 = (x2 - pad_w) / r
                    y2 = (y2 - pad_h) / r

                    # 裁剪到图像边界
                    x1 = max(0, min(orig_w, x1))
                    y1 = max(0, min(orig_h, y1))
                    x2 = max(0, min(orig_w, x2))
                    y2 = max(0, min(orig_h, y2))

                    if x1 >= x2 or y1 >= y2:
                        continue

                    cls = int(cls_id[i])
                    if cls < 0 or cls >= len(CLASS_NAMES):
                        continue

                    valid_boxes.append([x1, y1, x2, y2, max_conf[i], cls])

            # NMS
            return self._non_max_suppression(valid_boxes)

        except Exception as e:
            print(f"Postprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _non_max_suppression(self, boxes: List[List[float]]) -> List[List[float]]:
        if len(boxes) == 0:
            return []
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep = []
        while boxes:
            current = boxes.pop(0)
            keep.append(current)
            boxes = [box for box in boxes
                     if box[5] != current[5] or self._iou(current[:4], box[:4]) < IOU_THRESHOLD]
        return keep

    def _iou(self, box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)

    def _draw_boxes(self, image: np.ndarray, detections: List[List[float]]) -> np.ndarray:
        """改进的绘制方法，增加安全性检查"""
        for box in detections:
            try:
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)
                
                # 验证参数
                if (cls_id < 0 or cls_id >= len(CLASS_NAMES) or 
                    x1 >= x2 or y1 >= y2 or
                    conf < 0 or conf > 1):
                    continue
                
                # 转换为整数坐标
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # 生成颜色
                color = (0, (cls_id * 50) % 256, (cls_id * 100) % 256)
                
                # 绘制
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            except Exception as e:
                print(f"Error drawing box: {e}")
                continue
                
        return image

    def destroy(self):
        try:
            if hasattr(self, 'ctx'):
                self.ctx.pop()
                del self.ctx
            if hasattr(self, 'stream'):
                del self.stream
            if hasattr(self, 'bindings'):
                del self.bindings
            print("Resources released.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    detector = None
    try:
        engine_path = "/home/bpz/yolo系列网课/tensorrtx-yolov13/yolov13.engine"
        image_path = "/home/bpz/yolo系列网课/tensorrtx-yolov13/test_cup.jpg"

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        detector = YOLOv13TRT(engine_path)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")

        print(f"Image loaded: {image_path}, shape: {img.shape}")

        result = detector.infer(img)
        if result is not None:
            cv2.imwrite("result.jpg", result)
            print("Result saved to result.jpg")
            cv2.imshow("Detection Result", result)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Inference failed.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if detector is not None:
            detector.destroy()

if __name__ == "__main__":
    main()
