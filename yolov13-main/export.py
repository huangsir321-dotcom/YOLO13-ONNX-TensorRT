from ultralytics import YOLO
model = YOLO('/home/bpz/yolo系列网课/yolov13-main/runs/train/exp3/weights/best.pt')  # Replace with the desired model scale
model.export(format="onnx", half=True)