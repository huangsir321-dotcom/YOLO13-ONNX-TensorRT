
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model='/home/bpz/yolo系列网课/yolov13-main/ultralytics/cfg/models/v13/yolov13.yaml')
    model.load('/home/bpz/yolo系列网课/yolov13-main/yolov13n.pt')  
    # 一般情况下不加载预训练权重。
    model.train(data=r'/home/bpz/yolo系列网课/yolov13-main/dataset.yaml',
                imgsz=224,
                epochs=30,
                batch=1,
                workers=0,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                amp=False 
                )
