from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO(model='/home/bpz/yolo系列网课/yolov13-main/runs/train/exp4/weights/best.pt')  
    model.val(data='/home/bpz/yolo系列网课/yolov13-main/dataset.yaml')
