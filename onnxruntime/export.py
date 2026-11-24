from ultralytics import YOLO
model = YOLO('/home/bpz/yolo系列网课/onnxruntime/yolov13n.pt')  # Replace with the desired model scale
model.export(format="onnx", half=True)  
