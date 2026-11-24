import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple, Optional
import time

class YOLOv13TRT:
    def __init__(self, engine_path: str, class_names: List[str]):
        """
        初始化YOLOv13 TensorRT推理引擎
        
        参数:
            engine_path: TensorRT引擎文件路径
            class_names: 类别名称列表（必须与训练时完全一致，不允许为空）
        """
        if not isinstance(class_names, list) or not class_names:
            raise ValueError("class_names必须是非空的类别名称列表")

        # 配置参数
        self.conf_thresh = 0.5
        self.iou_threshold = 0.4
        self.input_width = 640
        self.input_height = 640
        self.class_names = class_names
        self.num_classes = len(self.class_names)

        # CUDA初始化
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()

        # TensorRT引擎初始化
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine.")
        
        self.context = self.engine.create_execution_context()

        # 分配内存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # 设置输入输出绑定
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
        print(f"Detecting {self.num_classes} classes: {self.class_names}")

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

            # 执行推理
            self.context.execute_async_v3(self.stream.handle)

            # 拷贝输出
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()

            # 3. 后处理
            output_data = self.outputs[0]['host']
            output_data = output_data.reshape(4 + self.num_classes, -1)  # (4 + num_classes, 8400)

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

        r = min(self.input_width / orig_w, self.input_height / orig_h)
        new_w = int(orig_w * r)
        new_h = int(orig_h * r)
        pad_w = (self.input_width - new_w) // 2
        pad_h = (self.input_height - new_h) // 2

        img = cv2.resize(img, (new_w, new_h))
        img = cv2.copyMakeBorder(
            img, pad_h, pad_h, pad_w, pad_w, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return np.ascontiguousarray(img), image, orig_h, orig_w

    def _postprocess(self, output: np.ndarray, orig_h: int, orig_w: int) -> List[List[float]]:
        try:
            if output.size == 0:
                return []

            num_boxes = output.shape[1]
            x_c = output[0, :]  # 中心x
            y_c = output[1, :]  # 中心y
            w = output[2, :]    # 宽
            h = output[3, :]    # 高
            conf = output[4:4 + self.num_classes, :]  # (num_classes, 8400)

            max_conf = np.max(conf, axis=0)  # (8400,)
            cls_id = np.argmax(conf, axis=0)  # (8400,) -> 0,1,2,...

            keep = max_conf > self.conf_thresh
            if not np.any(keep):
                return []

            r = min(self.input_width / orig_w, self.input_height / orig_h)
            pad_w = (self.input_width - r * orig_w) / 2
            pad_h = (self.input_height - r * orig_h) / 2

            valid_boxes = []
            for i in range(num_boxes):
                if keep[i]:
                    x1 = x_c[i] - w[i] / 2
                    y1 = y_c[i] - h[i] / 2
                    x2 = x_c[i] + w[i] / 2
                    y2 = y_c[i] + h[i] / 2

                    x1 = (x1 - pad_w) / r
                    y1 = (y1 - pad_h) / r
                    x2 = (x2 - pad_w) / r
                    y2 = (y2 - pad_h) / r

                    x1 = max(0, min(orig_w, x1))
                    y1 = max(0, min(orig_h, y1))
                    x2 = max(0, min(orig_w, x2))
                    y2 = max(0, min(orig_h, y2))

                    if x1 >= x2 or y1 >= y2:
                        continue

                    cls = int(cls_id[i])
                    if cls < 0 or cls >= self.num_classes:
                        continue

                    valid_boxes.append([x1, y1, x2, y2, max_conf[i], cls])

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
                     if box[5] != current[5] or self._iou(current[:4], box[:4]) < self.iou_threshold]
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
        for box in detections:
            try:
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)
                
                if (cls_id < 0 or cls_id >= self.num_classes or 
                    x1 >= x2 or y1 >= y2 or conf < 0 or conf > 1):
                    continue
                
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                color = (0, (cls_id * 50) % 256, (cls_id * 100) % 256)
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{self.class_names[cls_id]}: {conf:.2f}"
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

    def set_conf_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.conf_thresh = max(0.0, min(1.0, threshold))
        
    def set_iou_threshold(self, threshold: float):
        """设置IOU阈值"""
        self.iou_threshold = max(0.0, min(1.0, threshold))


def list_cameras(max_cameras: int = 5):
    """列出可用摄像头"""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def main():
    # 在此处直接修改自己的类别列表
    custom_classes = ["cat", "cup", "dog"]  

    engine_path = "/home/bpz/yolo系列网课/tensorrtx-yolov13/yolov13.engine"  # 修改为自己的引擎路径
    if not os.path.exists(engine_path):
        print(f"[ERROR] Engine file not found: {engine_path}")
        return

    # 列出可用摄像头
    cams = list_cameras()
    if len(cams) == 0:
        print("[ERROR] No camera found!")
        return

    print(f"Available cameras: {cams}")
    cam_id = cams[0]  # 默认使用第一个
    if len(cams) > 1:
        print("Multiple cameras detected. Using the first one.")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_id}")
        return

    print(f"Camera {cam_id} opened. Press 'q' to quit.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 初始化检测器
    detector = YOLOv13TRT(engine_path, class_names=custom_classes)

    # 改进的FPS计算
    prev_frame_time = 0
    new_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # 记录推理开始时间
            start_time = time.time()

            # 推理
            result_frame = detector.infer(frame)
            if result_frame is None:
                result_frame = frame.copy()

            # 计算FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - start_time)
            prev_frame_time = new_frame_time

            # 显示FPS（改进显示方式）
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                        font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 显示推理时间
            inference_time = (new_frame_time - start_time) * 1000  # 转换为毫秒
            cv2.putText(result_frame, f"Inference: {inference_time:.1f}ms", (10, 60),
                        font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("YOLOv13 TensorRT - Camera (q to quit)", result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q或ESC退出
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        cap.release()
        detector.destroy()
        cv2.destroyAllWindows()
        print("Camera and resources released.")


if __name__ == "__main__":
    main()
