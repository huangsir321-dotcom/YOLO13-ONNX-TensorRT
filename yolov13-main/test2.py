# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import YOLO

# 清理GPU缓存
torch.cuda.empty_cache()

if __name__ == '__main__':
    model = YOLO(model='/home/bpz/yolo系列网课/yolov13-main/ultralytics/cfg/models/v13/yolov13.yaml')
    model.load('/home/bpz/yolo系列网课/yolov13-main/yolov13n.pt')
    
    # 将模型设置为训练模式并移动到GPU
    model.model.train().cuda()
    
    # 创建一个假的输入数据（模拟一个batch）
    fake_input = torch.randn(1, 3, 224, 224).cuda()
    
    try:
        # 尝试前向传播
        with torch.no_grad():
            output = model.model(fake_input)
        print("Forward pass on GPU successful")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 如果前向传播成功，再尝试训练
    model.train(
        data='/home/bpz/yolo系列网课/yolov13-main/dataset.yaml',
        imgsz=224,
        epochs=30,
        batch=1,
        workers=0,
        device='0',
        optimizer='SGD',
        close_mosaic=10,
        amp=False,
        project='runs/train',
        name='exp'
    )
