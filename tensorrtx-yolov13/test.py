import tensorrt as trt

# 1. 加载引擎文件
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("yolov13.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 2. 检查输出层维度
for i in range(engine.num_bindings):
    if engine.binding_is_input(i):
        continue
    shape = engine.get_binding_shape(i)
    # YOLO输出通常是[bs, num_anchors, num_classes + 5], 最后一维的"5"是xywh+conf
    num_classes = shape[-1] - 5  # 假设输出格式符合YOLO标准
    print(f"Output layer {i}: shape={shape}, inferred num_classes={num_classes}")
    
