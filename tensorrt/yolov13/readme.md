## Introduction

Yolo12 model supports TensorRT-8.

Training code [link](https://github.com/ultralytics/ultralytics/archive/refs/tags/v8.3.38.zip)

## Environment

* cuda 11.8
* cudnn 8.9.1.23
* tensorrt 8.6.1.6
* opencv 4.8.0
* ultralytics 8.3.0

## Support

* [x] YOLO12-det support FP32/FP16 and C++ API


## Config

* Choose the YOLO12 sub-model n/s/m/l/x from command line arguments.
* Other configs please check [src/config.h](src/config.h)

## Build and Run

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```shell
# Download ultralytics
wget https://github.com/ultralytics/ultralytics/archive/refs/tags/v8.3.119.zip -O ultralytics-8.3.119.zip
# Unzip ultralytics
unzip ultralytics-8.3.119.zip
cd ultralytics-8.3.119
# Download models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt -O yolo12n.pt # to download other models, replace 'yolo12n.pt' with 'yolo12s.pt', 'yolo12m.pt', 'yolo12l.pt' or 'yolo12x.pt'
# Generate .wts
cp [PATH-TO-TENSORRTX]/yolov12/gen_wts.py .
python gen_wts.py -w yolo12n.pt -o yolo12n.wts -t detect
# A file 'yolo12n.wts' will be generated.
```

2. build tensorrtx/yolov12 and run
```shell
cd [PATH-TO-TENSORRTX]/yolov12
mkdir build
cd build
cmake ..
make
```

### Detection
```shell
cp [PATH-TO-ultralytics]/yolo12n.wts .
# Build and serialize TensorRT engine
./yolo12_det -s yolo12n.wts yolo12n.engine [n/s/m/l/x]
# Run inference
./yolo12_det -d yolo12n.engine ../images [c/g]
./yolo13_det -d yolo12n.engine ../images [c/g]
# results saved in build directory
```

## More Information
See the readme in [home page.](https://github.com/wang-xinyu/tensorrtx)


我用yolov13训练自己的数据集，自定义类别，我使用的是best.pt转yolov13.wts转yolov13.engine的方式,我在进行tesnoerrt yolov13 部署时，yolov13里面包括include(block.h、config.h、cuda_utils.h、logging.h、macros.h、model.h、postprocess.h、preprocess.h、types.h、utils.h)、plugin(yololayer.cu、yololayer.h)、src(block.cpp、model.cpp、postprocess.cpp、postprocess.cu、preprocess.cu)、CMakeList.txt、yolo13_det.cpp

请您将修改后的model.cpp完整的给我
这是print(best.pt)权重信息：
 conda run --live-stream --name yolov13 python /home/bpz/yolo系列网课/yolov13-main/print_model.py
{'date': '2025-07-20T14:28:01.510350', 'version': '8.3.63', 'license': 'AGPL-3.0 (https://ultralytics.com/license)', 'docs': 'https://docs.ultralytics.com', 'epoch': -1, 'best_fitness': None, 'model': DetectionModel(
  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (1): Conv(
      (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=2, bias=False)
      (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (2): DSC3k2(
      (cv1): Conv(
        (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(48, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): ModuleList(
        (0): DSBottleneck(
          (cv1): DSConv(
            (dw): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
            (pw): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): DSConv(
            (dw): Conv2d(16, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=16, bias=False)
            (pw): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (3): Conv(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (4): DSC3k2(
      (cv1): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): ModuleList(
        (0): DSBottleneck(
          (cv1): DSConv(
            (dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): DSConv(
            (dw): Conv2d(32, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=32, bias=False)
            (pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (5): DSConv(
      (dw): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
      (pw): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (6): A2C2f(
      (cv1): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): ModuleList(
        (0-1): 2 x Sequential(
          (0): ABlock(
            (attn): AAttn(
              (qk): Conv(
                (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (v): Conv(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (proj): Conv(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (pe): Conv(
                (conv): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
            (mlp): Sequential(
              (0): Conv(
                (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (1): Conv(
                (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
          )
          (1): ABlock(
            (attn): AAttn(
              (qk): Conv(
                (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (v): Conv(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (proj): Conv(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (pe): Conv(
                (conv): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=64, bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
            (mlp): Sequential(
              (0): Conv(
                (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (1): Conv(
                (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
          )
        )
      )
    )
    (7): DSConv(
      (dw): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
      (pw): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (8): A2C2f(
      (cv1): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): ModuleList(
        (0-1): 2 x Sequential(
          (0): ABlock(
            (attn): AAttn(
              (qk): Conv(
                (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (v): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (proj): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (pe): Conv(
                (conv): Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
            (mlp): Sequential(
              (0): Conv(
                (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (1): Conv(
                (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
          )
          (1): ABlock(
            (attn): AAttn(
              (qk): Conv(
                (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (v): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (proj): Conv(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (pe): Conv(
                (conv): Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
            (mlp): Sequential(
              (0): Conv(
                (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (1): Conv(
                (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
          )
        )
      )
    )
    (9): HyperACE(
      (cv1): Conv(
        (conv): Conv2d(128, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): ModuleList(
        (0): DSC3k(
          (cv1): Conv(
            (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv3): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (m): Sequential(
            (0): DSBottleneck(
              (cv1): DSConv(
                (dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                (pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (cv2): DSConv(
                (dw): Conv2d(32, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=32, bias=False)
                (pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
            (1): DSBottleneck(
              (cv1): DSConv(
                (dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                (pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (cv2): DSConv(
                (dw): Conv2d(32, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=32, bias=False)
                (pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
          )
        )
      )
      (fuse): FuseModule(
        (downsample): AvgPool2d(kernel_size=2, stride=2, padding=0)
        (upsample): Upsample(scale_factor=2.0, mode='nearest')
        (conv_out): Conv(
          (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
      )
      (branch1): C3AH(
        (cv1): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): AdaHGComputation(
          (hgnn): AdaHGConv(
            (edge_generator): AdaHyperedgeGen(
              (context_net): Linear(in_features=128, out_features=256, bias=True)
              (pre_head_proj): Linear(in_features=64, out_features=64, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (edge_proj): Sequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): GELU(approximate='none')
            )
            (node_proj): Sequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): GELU(approximate='none')
            )
          )
        )
        (cv3): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
      )
      (branch2): C3AH(
        (cv1): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): AdaHGComputation(
          (hgnn): AdaHGConv(
            (edge_generator): AdaHyperedgeGen(
              (context_net): Linear(in_features=128, out_features=256, bias=True)
              (pre_head_proj): Linear(in_features=64, out_features=64, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (edge_proj): Sequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): GELU(approximate='none')
            )
            (node_proj): Sequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): GELU(approximate='none')
            )
          )
        )
        (cv3): Conv(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
      )
    )
    (10): Upsample(scale_factor=2.0, mode='nearest')
    (11): DownsampleConv(
      (downsample): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (channel_adjust): Conv(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (12): FullPAD_Tunnel()
    (13): FullPAD_Tunnel()
    (14): FullPAD_Tunnel()
    (15): Upsample(scale_factor=2.0, mode='nearest')
    (16): Concat()
    (17): DSC3k2(
      (cv1): Conv(
        (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): ModuleList(
        (0): DSC3k(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv3): Conv(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (m): Sequential(
            (0): DSBottleneck(
              (cv1): DSConv(
                (dw): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (pw): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (cv2): DSConv(
                (dw): Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=64, bias=False)
                (pw): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
            (1): DSBottleneck(
              (cv1): DSConv(
                (dw): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (pw): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (cv2): DSConv(
                (dw): Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=64, bias=False)
                (pw): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
          )
        )
      )
    )
    (18): FullPAD_Tunnel()
    (19): Upsample(scale_factor=2.0, mode='nearest')
    (20): Concat()
    (21): DSC3k2(
      (cv1): Conv(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): ModuleList(
        (0): DSC3k(
          (cv1): Conv(
            (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv3): Conv(
            (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (m): Sequential(
            (0): DSBottleneck(
              (cv1): DSConv(
                (dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                (pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (cv2): DSConv(
                (dw): Conv2d(32, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=32, bias=False)
                (pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
            (1): DSBottleneck(
              (cv1): DSConv(
                (dw): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                (pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (cv2): DSConv(
                (dw): Conv2d(32, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=32, bias=False)
                (pw): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
          )
        )
      )
    )
    (22): Conv(
      (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (23): FullPAD_Tunnel()
    (24): Conv(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (25): Concat()
    (26): DSC3k2(
      (cv1): Conv(
        (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): ModuleList(
        (0): DSC3k(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv3): Conv(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (m): Sequential(
            (0): DSBottleneck(
              (cv1): DSConv(
                (dw): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (pw): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (cv2): DSConv(
                (dw): Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=64, bias=False)
                (pw): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
            (1): DSBottleneck(
              (cv1): DSConv(
                (dw): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (pw): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (cv2): DSConv(
                (dw): Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=64, bias=False)
                (pw): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
          )
        )
      )
    )
    (27): FullPAD_Tunnel()
    (28): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (29): Concat()
    (30): DSC3k2(
      (cv1): Conv(
        (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): ModuleList(
        (0): DSC3k(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv3): Conv(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (m): Sequential(
            (0): DSBottleneck(
              (cv1): DSConv(
                (dw): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (pw): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (cv2): DSConv(
                (dw): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=128, bias=False)
                (pw): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
            (1): DSBottleneck(
              (cv1): DSConv(
                (dw): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
                (pw): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (cv2): DSConv(
                (dw): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=128, bias=False)
                (pw): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
          )
        )
      )
    )
    (31): FullPAD_Tunnel()
    (32): Detect(
      (cv2): ModuleList(
        (0): Sequential(
          (0): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Sequential(
          (0): Conv(
            (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Sequential(
          (0): Conv(
            (conv): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (cv3): ModuleList(
        (0): Sequential(
          (0): Sequential(
            (0): DWConv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
          )
          (1): Sequential(
            (0): DWConv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
          )
          (2): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Sequential(
          (0): Sequential(
            (0): DWConv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (1): Conv(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
          )
          (1): Sequential(
            (0): DWConv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
          )
          (2): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Sequential(
          (0): Sequential(
            (0): DWConv(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (1): Conv(
              (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
          )
          (1): Sequential(
            (0): DWConv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
          )
          (2): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (dfl): DFL(
        (conv): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
    )
  )
), 'ema': None, 'updates': None, 'optimizer': None, 'train_args': {'task': 'detect', 'mode': 'train', 'model': '/home/bpz/yolo系列网课/yolov13-main/ultralytics/cfg/models/v13/yolov13.yaml', 'data': '/home/bpz/yolo系列网课/yolov13-main/dataset.yaml', 'epochs': 30, 'time': None, 'patience': 100, 'batch': 4, 'imgsz': 640, 'save': True, 'save_period': -1, 'cache': False, 'device': '0', 'workers': 2, 'project': 'runs/train', 'name': 'exp3', 'exist_ok': False, 'pretrained': False, 'optimizer': 'SGD', 'verbose': True, 'seed': 0, 'deterministic': True, 'single_cls': False, 'rect': False, 'cos_lr': False, 'close_mosaic': 10, 'resume': False, 'amp': True, 'fraction': 1.0, 'profile': False, 'freeze': None, 'multi_scale': False, 'overlap_mask': True, 'mask_ratio': 4, 'dropout': 0.0, 'val': True, 'split': 'val', 'save_json': False, 'save_hybrid': False, 'conf': None, 'iou': 0.7, 'max_det': 300, 'half': False, 'dnn': False, 'plots': True, 'source': None, 'vid_stride': 1, 'stream_buffer': False, 'visualize': False, 'augment': False, 'agnostic_nms': False, 'classes': None, 'retina_masks': False, 'embed': None, 'show': False, 'save_frames': False, 'save_txt': False, 'save_conf': False, 'save_crop': False, 'show_labels': True, 'show_conf': True, 'show_boxes': True, 'line_width': None, 'format': 'torchscript', 'keras': False, 'optimize': False, 'int8': False, 'dynamic': False, 'simplify': True, 'opset': None, 'workspace': None, 'nms': False, 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.0, 'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'bgr': 0.0, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.1, 'copy_paste_mode': 'flip', 'auto_augment': 'randaugment', 'erasing': 0.4, 'crop_fraction': 1.0, 'cfg': None, 'tracker': 'botsort.yaml'}, 'train_metrics': {'metrics/precision(B)': 0.74195, 'metrics/recall(B)': 0.75, 'metrics/mAP50(B)': 0.86246, 'metrics/mAP75(B)': 0.74568, 'metrics/mAP50-95(B)': 0.6687, 'val/box_loss': 1.1686, 'val/cls_loss': 2.37283, 'val/dfl_loss': 1.66939, 'fitness': 0.6687}, 'train_results': {'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 'time': [2.65096, 4.9663, 7.07088, 9.1539, 11.2305, 13.3, 15.2925, 17.357, 19.4035, 21.4379, 23.49, 25.5349, 27.5809, 29.6325, 31.6253, 33.6237, 35.6347, 37.6421, 39.6605, 41.6955, 43.9051, 46.0161, 48.0051, 50.0162, 52.0251, 54.0602, 56.0422, 58.0616, 60.1085, 62.1388], 'train/box_loss': [0.99869, 0.81929, 0.74423, 0.78372, 0.78623, 0.83285, 0.84895, 0.84657, 0.8715, 0.82019, 0.8191, 0.7772, 0.8464, 0.89491, 0.87384, 0.78956, 0.79153, 0.83879, 0.85001, 0.80666, 0.69586, 0.6744, 0.69515, 0.6686, 0.66049, 0.61811, 0.59399, 0.63406, 0.57151, 0.60316], 'train/cls_loss': [3.22314, 3.14698, 3.02646, 3.06383, 2.94228, 2.88239, 2.75749, 2.603, 2.62163, 2.3328, 2.14844, 2.1015, 2.08904, 2.14125, 2.11745, 2.07203, 1.90562, 1.94941, 1.76479, 1.87384, 2.43669, 2.41524, 2.38492, 2.47908, 2.26157, 2.2351, 2.15632, 2.23139, 2.26414, 2.21105], 'train/dfl_loss': [1.49171, 1.35375, 1.31031, 1.35836, 1.38005, 1.41005, 1.38407, 1.39205, 1.3895, 1.392, 1.36153, 1.37127, 1.37227, 1.4341, 1.36527, 1.33699, 1.37598, 1.45892, 1.35144, 1.40945, 1.3069, 1.20667, 1.35029, 1.31234, 1.32493, 1.20079, 1.13758, 1.26791, 1.21612, 1.20217], 'metrics/precision(B)': [0.00542, 0.00555, 0.0055, 0.00538, 0.00489, 0.00475, 0.00475, 0.00461, 0.80013, 0.35561, 0.35561, 0.3954, 0.33813, 0.50118, 0.50118, 0.40771, 0.47948, 0.50079, 0.50079, 0.51375, 0.64008, 0.57171, 0.57171, 0.53417, 0.56347, 0.6165, 0.6165, 0.68324, 0.74195, 0.79261], 'metrics/recall(B)': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.42785, 0.75635, 0.75635, 0.83333, 0.83333, 0.83333, 0.83333, 0.89343, 0.81009, 0.83333, 0.83333, 0.83333, 0.78277, 0.90144, 0.90144, 0.91667, 0.74851, 0.75, 0.75, 0.74562, 0.75, 0.75], 'metrics/mAP50(B)': [0.14087, 0.25722, 0.37521, 0.48856, 0.54154, 0.54244, 0.54244, 0.52171, 0.56583, 0.47454, 0.47454, 0.46199, 0.56589, 0.79696, 0.79696, 0.55861, 0.585, 0.64239, 0.64239, 0.63673, 0.70475, 0.6303, 0.6303, 0.59598, 0.54882, 0.63398, 0.63398, 0.82501, 0.86246, 0.84115], 'metrics/mAP75(B)': [0.05394, 0.17168, 0.30594, 0.43464, 0.49427, 0.49988, 0.49988, 0.47946, 0.48095, 0.42671, 0.42671, 0.43515, 0.50479, 0.78687, 0.78687, 0.52425, 0.46782, 0.47243, 0.47243, 0.45055, 0.52771, 0.45872, 0.45872, 0.4628, 0.48321, 0.57932, 0.57932, 0.7477, 0.74568, 0.74571], 'metrics/mAP50-95(B)': [0.07137, 0.15997, 0.26683, 0.36879, 0.40779, 0.39185, 0.39185, 0.41943, 0.41602, 0.37134, 0.37134, 0.35165, 0.42947, 0.5734, 0.5734, 0.4232, 0.39266, 0.42239, 0.42239, 0.45408, 0.49776, 0.42198, 0.42198, 0.42615, 0.39536, 0.48052, 0.48052, 0.65564, 0.6687, 0.66531], 'val/box_loss': [0.87382, 0.82265, 0.82234, 0.85081, 0.95403, 1.00942, 1.00942, 0.91154, 1.01859, 1.08233, 1.08233, 1.37343, 1.38089, 1.29276, 1.29276, 1.33916, 1.19729, 1.1977, 1.1977, 1.15265, 1.18159, 1.17313, 1.17313, 1.26595, 1.38998, 1.37096, 1.37096, 1.20666, 1.1686, 1.17009], 'val/cls_loss': [3.32795, 3.3788, 3.37265, 3.38431, 3.30812, 3.26023, 3.26023, 3.1258, 2.95279, 2.95236, 2.95236, 3.38045, 2.77482, 2.52694, 2.52694, 2.32515, 2.44157, 2.4055, 2.4055, 2.34089, 2.42843, 2.47294, 2.47294, 2.55882, 2.63514, 2.42054, 2.42054, 2.42258, 2.37283, 2.31844], 'val/dfl_loss': [1.53707, 1.48056, 1.45778, 1.42292, 1.45392, 1.51492, 1.51492, 1.44912, 1.69648, 1.86165, 1.86165, 2.18766, 2.15115, 2.00164, 2.00164, 1.95097, 1.82803, 1.77869, 1.77869, 1.74014, 1.76609, 1.68702, 1.68702, 1.78904, 1.91448, 1.86412, 1.86412, 1.75671, 1.66939, 1.67098], 'lr/pg0': [0.0011, 0.0022241, 0.003269, 0.0042347, 0.0051212, 0.0059285, 0.0066566, 0.0073055, 0.00736, 0.00703, 0.0067, 0.00637, 0.00604, 0.00571, 0.00538, 0.00505, 0.00472, 0.00439, 0.00406, 0.00373, 0.0034, 0.00307, 0.00274, 0.00241, 0.00208, 0.00175, 0.00142, 0.00109, 0.00076, 0.00043], 'lr/pg1': [0.0011, 0.0022241, 0.003269, 0.0042347, 0.0051212, 0.0059285, 0.0066566, 0.0073055, 0.00736, 0.00703, 0.0067, 0.00637, 0.00604, 0.00571, 0.00538, 0.00505, 0.00472, 0.00439, 0.00406, 0.00373, 0.0034, 0.00307, 0.00274, 0.00241, 0.00208, 0.00175, 0.00142, 0.00109, 0.00076, 0.00043], 'lr/pg2': [0.0011, 0.0022241, 0.003269, 0.0042347, 0.0051212, 0.0059285, 0.0066566, 0.0073055, 0.00736, 0.00703, 0.0067, 0.00637, 0.00604, 0.00571, 0.00538, 0.00505, 0.00472, 0.00439, 0.00406, 0.00373, 0.0034, 0.00307, 0.00274, 0.00241, 0.00208, 0.00175, 0.00142, 0.00109, 0.00076, 0.00043]}}

这是model.cpp内容：
#include <math.h>
#include <iostream>

#include "block.h"
//#include "calibrator.h"
#include "config.h"
#include "model.h"

static int get_width(int x, float gw, int max_channels, int divisor = 8) {
    auto channel = std::min(x, max_channels);
    channel = int(ceil((channel * gw) / divisor)) * divisor;
    return channel;
}

static int get_depth(int x, float gd) {
    if (x == 1)
        return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0)
        --r;
    return std::max<int>(r, 1);
}

void calculateStrides(nvinfer1::IElementWiseLayer* conv_layers[], int size, int reference_size, int strides[]) {
    for (int i = 0; i < size; ++i) {
        nvinfer1::ILayer* layer = conv_layers[i];
        nvinfer1::Dims dims = layer->getOutput(0)->getDimensions();
        int feature_map_size = dims.d[2];
        strides[i] = reference_size / feature_map_size;
    }
}

nvinfer1::IHostMemory* buildEngineYolo12Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type)

{
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);


    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), {3, 3}, 2, "model.0");

    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0),
                                                    get_width(128, gw, max_channels), {3, 3}, 2, "model.1");

    bool c3k = false;
    if (type == "m" || type == "l" || type == "x") {
        c3k = true;
    }

    nvinfer1::IElementWiseLayer* conv2 =
            C3K2(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.2");

    nvinfer1::IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0),
                                                    get_width(256, gw, max_channels), {3, 3}, 2, "model.3");

    nvinfer1::IElementWiseLayer* conv4 =
            C3K2(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.4");

    nvinfer1::IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0),
                                                    get_width(512, gw, max_channels), {3, 3}, 2, "model.5");

    nvinfer1::ILayer* conv6 = A2C2f(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                                    get_width(512, gw, max_channels), 4, true, 4, true, 2.0, 0.25, 1, true, "model.6");

    nvinfer1::IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0),
                                                    get_width(1024, gw, max_channels), {3, 3}, 2, "model.7");

    nvinfer1::ILayer* conv8 = A2C2f(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                                    get_width(1024, gw, max_channels), 4, true, 1, true, 2.0, 0.25, 1, true, "model.8");

 
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample9 = network->addResize(*conv8->getOutput(0));
    assert(upsample9);
    upsample9->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample9->setScales(scale, 4);

    nvinfer1::ITensor* inputTensors10[] = {upsample9->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat10 = network->addConcatenation(inputTensors10, 2);

    nvinfer1::ILayer* conv11 =
            A2C2f(network, weightMap, *cat10->getOutput(0), get_width(1024, gw, max_channels),
                  get_width(512, gw, max_channels), 4, false, 1, true, 2.0, 0.25, 1, true, "model.11");

    nvinfer1::IResizeLayer* upsample12 = network->addResize(*conv11->getOutput(0));
    assert(upsample12);
    upsample12->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample12->setScales(scale, 4);
    nvinfer1::ITensor* inputTensors13[] = {upsample12->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat13 = network->addConcatenation(inputTensors13, 2);
    nvinfer1::ILayer* conv14 =
            A2C2f(network, weightMap, *cat13->getOutput(0), get_width(256, gw, max_channels),
                  get_width(256, gw, max_channels), 4, false, 1, true, 2.0, 0.25, 1, true, "model.14");

    nvinfer1::IElementWiseLayer* conv15 = convBnSiLU(network, weightMap, *conv14->getOutput(0),
                                                     get_width(256, gw, max_channels), {3, 3}, 2, "model.15");
    nvinfer1::ITensor* inputTensors16[] = {conv15->getOutput(0), conv11->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat16 = network->addConcatenation(inputTensors16, 2);

    nvinfer1::ILayer* conv17 =
            A2C2f(network, weightMap, *cat16->getOutput(0), get_width(512, gw, max_channels),
                  get_width(512, gw, max_channels), 4, false, 1, true, 2.0, 0.25, 1, true, "model.17");

    nvinfer1::IElementWiseLayer* conv18 = convBnSiLU(network, weightMap, *conv17->getOutput(0),
                                                     get_width(512, gw, max_channels), {3, 3}, 2, "model.18");
    nvinfer1::ITensor* inputTensors19[] = {conv18->getOutput(0), conv8->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat19 = network->addConcatenation(inputTensors19, 2);

    nvinfer1::IElementWiseLayer* conv20 =
            C3K2(network, weightMap, *cat19->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.20");

    int c2 = std::max(std::max(16, get_width(256, gw, max_channels) / 4), 16 * 4);
    int c3 = std::max(get_width(256, gw, max_channels), std::min(kNumClass, 100));

    // output 0
    nvinfer1::IElementWiseLayer* conv21_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv14->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv21_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv21_cv2_0_0->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.0.1");

    nvinfer1::IConvolutionLayer* conv21_cv2_0_2 =
            network->addConvolutionNd(*conv21_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.21.cv2.0.2.weight"], weightMap["model.21.cv2.0.2.bias"]);
    conv21_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});

    auto* conv21_cv3_0_0_0 = DWConv(network, weightMap, *conv14->getOutput(0), get_width(256, gw, max_channels), {3, 3},
                                    1, "model.21.cv3.0.0.0");
    auto* conv21_cv3_0_0_1 =
            convBnSiLU(network, weightMap, *conv21_cv3_0_0_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.0.0.1");
    auto* conv21_cv3_0_1_0 =
            DWConv(network, weightMap, *conv21_cv3_0_0_1->getOutput(0), c3, {3, 3}, 1, "model.21.cv3.0.1.0");
    auto* conv21_cv3_0_1_1 =
            convBnSiLU(network, weightMap, *conv21_cv3_0_1_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.0.1.1");
    nvinfer1::IConvolutionLayer* conv21_cv3_0_2 =
            network->addConvolutionNd(*conv21_cv3_0_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.21.cv3.0.2.weight"], weightMap["model.21.cv3.0.2.bias"]);
    conv21_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});

    nvinfer1::ITensor* inputTensor21_0[] = {conv21_cv2_0_2->getOutput(0), conv21_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21_0 = network->addConcatenation(inputTensor21_0, 2);

    //output 1
    nvinfer1::IElementWiseLayer* conv21_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv17->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv21_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv21_cv2_1_0->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv21_cv2_1_2 =
            network->addConvolutionNd(*conv21_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.21.cv2.1.2.weight"], weightMap["model.21.cv2.1.2.bias"]);
    conv21_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv21_cv3_1_0_0 = DWConv(network, weightMap, *conv17->getOutput(0), get_width(512, gw, max_channels), {3, 3},
                                    1, "model.21.cv3.1.0.0");
    auto* conv21_cv3_1_0_1 =
            convBnSiLU(network, weightMap, *conv21_cv3_1_0_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.1.0.1");
    auto* conv21_cv3_1_1_0 =
            DWConv(network, weightMap, *conv21_cv3_1_0_1->getOutput(0), c3, {3, 3}, 1, "model.21.cv3.1.1.0");
    auto* conv21_cv3_1_1_1 =
            convBnSiLU(network, weightMap, *conv21_cv3_1_1_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.1.1.1");
    nvinfer1::IConvolutionLayer* conv21_cv3_1_2 =
            network->addConvolutionNd(*conv21_cv3_1_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.21.cv3.1.2.weight"], weightMap["model.21.cv3.1.2.bias"]);
    conv21_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor21_1[] = {conv21_cv2_1_2->getOutput(0), conv21_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21_1 = network->addConcatenation(inputTensor21_1, 2);

    //output 2
    nvinfer1::IElementWiseLayer* conv21_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv20->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv21_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv21_cv2_2_0->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv21_cv2_2_2 =
            network->addConvolutionNd(*conv21_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.21.cv2.2.2.weight"], weightMap["model.21.cv2.2.2.bias"]);
    conv21_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv21_cv3_2_0_0 = DWConv(network, weightMap, *conv20->getOutput(0), get_width(1024, gw, max_channels),
                                    {3, 3}, 1, "model.21.cv3.2.0.0");
    auto* conv21_cv3_2_0_1 =
            convBnSiLU(network, weightMap, *conv21_cv3_2_0_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.2.0.1");
    auto* conv21_cv3_2_1_0 =
            DWConv(network, weightMap, *conv21_cv3_2_0_1->getOutput(0), c3, {3, 3}, 1, "model.21.cv3.2.1.0");
    auto* conv21_cv3_2_1_1 =
            convBnSiLU(network, weightMap, *conv21_cv3_2_1_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.2.1.1");
    nvinfer1::IConvolutionLayer* conv21_cv3_2_2 =
            network->addConvolutionNd(*conv21_cv3_2_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.21.cv3.2.2.weight"], weightMap["model.21.cv3.2.2.bias"]);
    conv21_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor21_2[] = {conv21_cv2_2_2->getOutput(0), conv21_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21_2 = network->addConcatenation(inputTensor21_2, 2);
    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer* shuffle21_0 = network->addShuffle(*cat21_0->getOutput(0));
    shuffle21_0->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split21_0_0 = network->addSlice(
            *shuffle21_0->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split21_0_1 =
            network->addSlice(*shuffle21_0->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])},
                              nvinfer1::Dims3{1, 1, 1});

    nvinfer1::IShuffleLayer* dfl21_0 =
            DFL(network, weightMap, *split21_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.21.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor22_dfl_0[] = {dfl21_0->getOutput(0), split21_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 2);
    cat22_dfl_0->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle21_1 = network->addShuffle(*cat21_1->getOutput(0));
    shuffle21_1->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split21_1_0 = network->addSlice(
            *shuffle21_1->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split21_1_1 =
            network->addSlice(*shuffle21_1->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl21_1 =
            DFL(network, weightMap, *split21_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.21.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor22_dfl_1[] = {dfl21_1->getOutput(0), split21_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 2);
    cat22_dfl_1->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle21_2 = network->addShuffle(*cat21_2->getOutput(0));
    shuffle21_2->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split21_2_0 = network->addSlice(
            *shuffle21_2->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split21_2_1 =
            network->addSlice(*shuffle21_2->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl21_2 =
            DFL(network, weightMap, *split21_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.21.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor22_dfl_2[] = {dfl21_2->getOutput(0), split21_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 2);
    cat22_dfl_2->setAxis(1);

    nvinfer1::IPluginV2Layer* yolo =
            addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer*>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2},
                         strides, stridesLength, false, false, false);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}

这是block.cpp内容：
#include "block.h"
#include <assert.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include "config.h"
#include "model.h"
#include "yololayer.h"

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> WeightMap;

    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; x++) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        WeightMap[name] = wt;
    }
    return WeightMap;
}

nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                      std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, pval, len};
    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    nvinfer1::IScaleLayer* output = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(output);
    return output;
}

nvinfer1::IElementWiseLayer* convBnSiLU(nvinfer1::INetworkDefinition* network,
                                        std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                        int ch, std::vector<int> k, int s, std::string lname) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k[0], k[1]},
                                                                  weightMap[lname + ".conv.weight"], bias_empty);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    // auto pad
    int p0 = k[0] / 2;
    int p1 = k[1] / 2;
    conv->setPaddingNd(nvinfer1::DimsHW{p0, p1});

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);

    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    nvinfer1::IElementWiseLayer* ew =
            network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

static nvinfer1::ILayer* bottleneck(nvinfer1::INetworkDefinition* network,
                                    std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                    int c1, int c2, bool shortcut, std::vector<int> k1, std::vector<int> k2, float e,
                                    std::string lname) {
    int c_ = (int)((float)c2 * e);
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, c_, k1, 1, lname + ".cv1");
    nvinfer1::IElementWiseLayer* conv2 =
            convBnSiLU(network, weightMap, *conv1->getOutput(0), c2, k2, 1, lname + ".cv2");

    if (shortcut && c1 == c2) {
        nvinfer1::IElementWiseLayer* ew =
                network->addElementWise(input, *conv2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        return ew;
    }
    return conv2;
}

nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c1,
                                  int c2, int k, std::string lname) {
    int c_ = c1 / 2;
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, c_, {1, 1}, 1, lname + ".cv1");
    nvinfer1::IPoolingLayer* pool1 =
            network->addPoolingNd(*conv1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k, k});
    pool1->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool1->setPaddingNd(nvinfer1::DimsHW{k / 2, k / 2});
    nvinfer1::IPoolingLayer* pool2 =
            network->addPoolingNd(*pool1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k, k});
    pool2->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool2->setPaddingNd(nvinfer1::DimsHW{k / 2, k / 2});
    nvinfer1::IPoolingLayer* pool3 =
            network->addPoolingNd(*pool2->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k, k});
    pool3->setStrideNd(nvinfer1::DimsHW{1, 1});
    pool3->setPaddingNd(nvinfer1::DimsHW{k / 2, k / 2});
    nvinfer1::ITensor* inputTensors[] = {conv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0),
                                         pool3->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensors, 4);
    nvinfer1::IElementWiseLayer* conv2 =
            convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1, 1}, 1, lname + ".cv2");
    return conv2;
}

nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                             nvinfer1::ITensor& input, int ch, int grid, int k, int s, int p, std::string lname) {

    nvinfer1::IShuffleLayer* shuffle1 = network->addShuffle(input);
    shuffle1->setReshapeDimensions(nvinfer1::Dims4{kBatchSize, 4, 16, grid});
    shuffle1->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*shuffle1->getOutput(0));
    softmax->setAxes(1 << 1);

    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv =
            network->addConvolutionNd(*softmax->getOutput(0), 1, nvinfer1::DimsHW{1, 1}, weightMap[lname], bias_empty);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setPaddingNd(nvinfer1::DimsHW{p, p});

    nvinfer1::IShuffleLayer* shuffle2 = network->addShuffle(*conv->getOutput(0));
    shuffle2->setReshapeDimensions(nvinfer1::Dims3{kBatchSize, 4, grid});

    return shuffle2;
}

nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition* network,
                                       std::vector<nvinfer1::IConcatenationLayer*> dets, const int* px_arry,
                                       int px_arry_num, bool is_segmentation, bool is_pose, bool is_obb) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const int netinfo_count = 9;  // Assuming the first 5 elements are for netinfo as per existing code.
    const int total_count = netinfo_count + px_arry_num;  // Total number of elements for netinfo and px_arry combined.

    std::vector<int> combinedInfo(total_count);
    int class_num = kNumClass;
    if (is_pose)
        class_num = kPoseNumClass;
    else if (is_obb)
        class_num = kObbNumClass;
    int input_w = kInputW;
    if (is_obb)
        input_w = kObbInputW;
    int input_h = kInputH;
    if (is_obb)
        input_h = kObbInputH;
    // Fill in the first 5 elements as per existing netinfo.
    combinedInfo[0] = class_num;
    combinedInfo[1] = kNumberOfPoints;
    combinedInfo[2] = kConfThreshKeypoints;
    combinedInfo[3] = input_w;
    combinedInfo[4] = input_h;
    combinedInfo[5] = kMaxNumOutputBbox;
    combinedInfo[6] = is_segmentation;
    combinedInfo[7] = is_pose;
    combinedInfo[8] = is_obb;

    // Copy the contents of px_arry into the combinedInfo vector after the initial 5 elements.
    std::copy(px_arry, px_arry + px_arry_num, combinedInfo.begin() + netinfo_count);

    // Now let's create the PluginField object to hold this combined information.
    nvinfer1::PluginField pluginField;
    pluginField.name = "combinedInfo";  // This can be any name that the plugin will recognize
    pluginField.data = combinedInfo.data();
    pluginField.type = nvinfer1::PluginFieldType::kINT32;
    pluginField.length = combinedInfo.size();

    // Create the PluginFieldCollection to hold the PluginField object.
    nvinfer1::PluginFieldCollection pluginFieldCollection;
    pluginFieldCollection.nbFields = 1;  // We have just one field, but it's a combined array
    pluginFieldCollection.fields = &pluginField;

    // Create the plugin object using the PluginFieldCollection.
    nvinfer1::IPluginV2* pluginObject = creator->createPlugin("yololayer", &pluginFieldCollection);

    // We assume that the plugin is to be added onto the network.
    // Prepare input tensors for the YOLO Layer.
    std::vector<nvinfer1::ITensor*> inputTensors;
    for (auto det : dets) {
        inputTensors.push_back(det->getOutput(0));  // Assuming each IConcatenationLayer has one output tensor.
    }

    // Add the plugin to the network using the prepared input tensors.
    nvinfer1::IPluginV2Layer* yoloLayer = network->addPluginV2(inputTensors.data(), inputTensors.size(), *pluginObject);

    return yoloLayer;  // Return the added YOLO layer.
}

static nvinfer1::ILayer* C3k(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                             nvinfer1::ITensor& input, int c1, int c2, int n, bool shortcut, std::vector<int> k1,
                             std::vector<int> k2, float e, std::string lname) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBnSiLU(network, weightMap, input, c_, {1, 1}, 1, lname + ".cv1");
    auto cv2 = convBnSiLU(network, weightMap, input, c_, {1, 1}, 1, lname + ".cv2");
    nvinfer1::ITensor* y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, k1, k2, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }

    nvinfer1::ITensor* inputTensors[] = {y1, cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1, 1}, 1, lname + ".cv3");
    return cv3;
}

nvinfer1::IElementWiseLayer* C3K2(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c1,
                                  int c2, int n, bool c3k, bool shortcut, float e, std::string lname) {
    int c_ = (float)c2 * e;

    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, 2 * c_, {1, 1}, 1, lname + ".cv1");
    nvinfer1::Dims d = conv1->getOutput(0)->getDimensions();

    nvinfer1::ISliceLayer* split1 =
            network->addSlice(*conv1->getOutput(0), nvinfer1::Dims4{0, 0, 0, 0},
                              nvinfer1::Dims4{d.d[0], d.d[1] / 2, d.d[2], d.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::ISliceLayer* split2 =
            network->addSlice(*conv1->getOutput(0), nvinfer1::Dims4{0, d.d[1] / 2, 0, 0},
                              nvinfer1::Dims4{d.d[0], d.d[1] / 2, d.d[2], d.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::ITensor* inputTensor0[] = {split1->getOutput(0), split2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensor0, 2);
    nvinfer1::ITensor* y1 = split2->getOutput(0);
    for (int i = 0; i < n; i++) {
        nvinfer1::ILayer* b;
        if (c3k) {
            b = C3k(network, weightMap, *y1, c_, c_, 2, shortcut, {3, 3}, {3, 3}, 0.5,
                    lname + ".m." + std::to_string(i));
        } else {
            b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, {3, 3}, {3, 3}, 0.5,
                           lname + ".m." + std::to_string(i));
        }
        y1 = b->getOutput(0);

        nvinfer1::ITensor* inputTensors[] = {cat->getOutput(0), b->getOutput(0)};
        cat = network->addConcatenation(inputTensors, 2);
    }

    nvinfer1::IElementWiseLayer* conv2 =
            convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1, 1}, 1, lname + ".cv2");

    return conv2;
}

static nvinfer1::ILayer* convBn(nvinfer1::INetworkDefinition* network,
                                std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int ch,
                                int k, int s, std::string lname, int g = 1) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv;
    if (lname.find(".pe") != std::string::npos) {
        nvinfer1::Weights conv_bias = weightMap[lname + ".conv.bias"];
        conv = network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k, k}, weightMap[lname + ".conv.weight"],
                                         conv_bias);
        assert(conv);
        conv->setStrideNd(nvinfer1::DimsHW{s, s});
        int p = k / 2;
        conv->setPaddingNd(nvinfer1::DimsHW{p, p});
        conv->setNbGroups(g);
        conv->setName((lname + ".conv").c_str());

        nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);
        bn->setName((lname + ".bn").c_str());
        return bn;

    } else {
        conv = network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k, k}, weightMap[lname + ".conv.weight"],
                                         bias_empty);
        assert(conv);
        conv->setStrideNd(nvinfer1::DimsHW{s, s});
        int p = k / 2;
        conv->setPaddingNd(nvinfer1::DimsHW{p, p});
        conv->setNbGroups(g);
        conv->setName((lname + ".conv").c_str());

        nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);
        bn->setName((lname + ".bn").c_str());
        return bn;
    }
}

static nvinfer1::ILayer* Attention(nvinfer1::INetworkDefinition* network,
                                   std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                   int dim, int num_heads, float attn_ratio, std::string lname) {
    int head_dim = dim / num_heads;
    int key_dim = head_dim * attn_ratio;
    float scale = pow(key_dim, -0.5);
    int nh_kd = key_dim * num_heads;
    int h = dim + nh_kd * 2;

    auto d = input.getDimensions();
    int B = d.d[0];
    int H = d.d[2];
    int W = d.d[3];
    int N = H * W;
    auto* qkv = convBn(network, weightMap, input, h, 1, 1, lname + ".qkv");
    // qkv.view(B, self.num_heads, -1, N)
    auto shuffle = network->addShuffle(*qkv->getOutput(0));
    shuffle->setReshapeDimensions(nvinfer1::Dims4{B, num_heads, -1, N});
    // q, k, v = .split([self.key_dim, self.key_dim, self.head_dim], dim=2)
    auto d1 = shuffle->getOutput(0)->getDimensions();
    auto q = network->addSlice(*shuffle->getOutput(0), nvinfer1::Dims4{0, 0, 0, 0},
                               nvinfer1::Dims4{d1.d[0], d1.d[1], key_dim, d1.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    auto k = network->addSlice(*shuffle->getOutput(0), nvinfer1::Dims4{0, 0, key_dim, 0},
                               nvinfer1::Dims4{d1.d[0], d1.d[1], key_dim, d1.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    auto v = network->addSlice(*shuffle->getOutput(0), nvinfer1::Dims4{0, 0, key_dim * 2, 0},
                               nvinfer1::Dims4{d1.d[0], d1.d[1], head_dim, d1.d[3]}, nvinfer1::Dims4{1, 1, 1, 1});
    // attn = ((q.transpose(-2, -1) @ k) * self.scale)
    auto qT = network->addShuffle(*q->getOutput(0));
    qT->setFirstTranspose(nvinfer1::Permutation{0, 1, 3, 2});
    auto matmul = network->addMatrixMultiply(*qT->getOutput(0), nvinfer1::MatrixOperation::kNONE, *k->getOutput(0),
                                             nvinfer1::MatrixOperation::kNONE);
    // There are not many memory leaks, and I will change it when I have time
    float* scale_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    scale_val[0] = scale;
    nvinfer1::Weights s_w{nvinfer1::DataType::kFLOAT, scale_val, 1};
    float* shift_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    shift_val[0] = 0;
    nvinfer1::Weights sh_w{nvinfer1::DataType::kFLOAT, shift_val, 1};
    float* power_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    power_val[0] = 1;
    nvinfer1::Weights p_w{nvinfer1::DataType::kFLOAT, power_val, 1};
    nvinfer1::IScaleLayer* scaleLayer =
            network->addScale(*matmul->getOutput(0), nvinfer1::ScaleMode::kUNIFORM, sh_w, s_w, p_w);
    // attn = attn.softmax(dim=-1)
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*scaleLayer->getOutput(0));
    softmax->setAxes(1 << 3);
    // x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W) + self.pe(v.reshape(B, -1, H, W))
    auto attnT = network->addShuffle(*softmax->getOutput(0));
    attnT->setFirstTranspose(nvinfer1::Permutation{0, 1, 3, 2});
    auto matmul2 = network->addMatrixMultiply(*v->getOutput(0), nvinfer1::MatrixOperation::kNONE, *attnT->getOutput(0),
                                              nvinfer1::MatrixOperation::kNONE);
    auto reshape = network->addShuffle(*matmul2->getOutput(0));
    reshape->setReshapeDimensions(nvinfer1::Dims4{B, -1, H, W});
    auto v_reshape = network->addShuffle(*v->getOutput(0));
    v_reshape->setReshapeDimensions(nvinfer1::Dims4{B, -1, H, W});
    // self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
    auto pe = convBn(network, weightMap, *v_reshape->getOutput(0), dim, 3, 1, lname + ".pe", dim);
    auto sum = network->addElementWise(*reshape->getOutput(0), *pe->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    // x = self.proj(x)
    // self.proj = Conv(dim, dim, 1, act=False)
    auto proj = convBn(network, weightMap, *sum->getOutput(0), dim, 1, 1, lname + ".proj");
    return proj;
}

static nvinfer1::ILayer* PSABlock(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int dim,
                                  float attn_ratio, int num_heads, bool shortcut, std::string lname) {
    // x = x + self.attn(x) if self.add else self.attn(x)
    auto attn = Attention(network, weightMap, input, dim, num_heads, attn_ratio, lname + ".attn");
    nvinfer1::ILayer* shortcut_layer = nullptr;
    if (shortcut) {
        shortcut_layer = network->addElementWise(input, *attn->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    } else {
        shortcut_layer = attn;
    }
    // self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
    // x = x + self.ffn(x) if self.add else self.ffn(x)
    auto ffn0 = convBnSiLU(network, weightMap, *shortcut_layer->getOutput(0), dim * 2, {1, 1}, 1, lname + ".ffn.0");
    auto ffn1 = convBn(network, weightMap, *ffn0->getOutput(0), dim, 1, 1, lname + ".ffn.1");
    if (shortcut) {
        return network->addElementWise(*shortcut_layer->getOutput(0), *ffn1->getOutput(0),
                                       nvinfer1::ElementWiseOperation::kSUM);
    } else {
        return ffn1;
    }
}

nvinfer1::ILayer* C2PSA(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap,
                        nvinfer1::ITensor& input, int c1, int c2, int n, float e, std::string lname) {
    assert(network != nullptr);
    int c = c1 * e;

    // cv1 branch
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, 2 * c, {1, 1}, 1, lname + ".cv1");
    nvinfer1::ITensor* cv1_out = conv1->getOutput(0);

    // Split the output of cv1 into two tensors
    nvinfer1::Dims dims = cv1_out->getDimensions();
    nvinfer1::ISliceLayer* split1 = network->addSlice(*cv1_out, nvinfer1::Dims4{0, 0, 0, 0},
                                                      nvinfer1::Dims4{dims.d[0], dims.d[1] / 2, dims.d[2], dims.d[3]},
                                                      nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::ISliceLayer* split2 = network->addSlice(*cv1_out, nvinfer1::Dims4{0, dims.d[1] / 2, 0, 0},
                                                      nvinfer1::Dims4{dims.d[0], dims.d[1] / 2, dims.d[2], dims.d[3]},
                                                      nvinfer1::Dims4{1, 1, 1, 1});

    // Create y1 bottleneck sequence
    nvinfer1::ITensor* y = split2->getOutput(0);
    for (int i = 0; i < n; ++i) {
        auto* bottleneck_layer =
                PSABlock(network, weightMap, *y, c, 0.5, c / 64, true, lname + ".m." + std::to_string(i));
        y = bottleneck_layer->getOutput(0);  // update 'y1' to be the output of the current bottleneck
    }

    // Concatenate y1 with the second split of cv1
    nvinfer1::ITensor* concatInputs[2] = {split1->getOutput(0), y};
    nvinfer1::IConcatenationLayer* cat = network->addConcatenation(concatInputs, 2);

    // cv2 to produce the final output
    nvinfer1::IElementWiseLayer* conv2 =
            convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1, 1}, 1, lname + ".cv2");

    return conv2;
}

nvinfer1::ILayer* DWConv(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                         nvinfer1::ITensor& input, int ch, std::vector<int> k, int s, std::string lname) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k[0], k[1]},
                                                                  weightMap[lname + ".conv.weight"], bias_empty);
    assert(conv);
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setNbGroups(ch);
    // auto pad
    int p0 = k[0] / 2;
    int p1 = k[1] / 2;
    conv->setPaddingNd(nvinfer1::DimsHW{p0, p1});

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);

    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    nvinfer1::IElementWiseLayer* ew =
            network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

nvinfer1::ILayer* A2C2f(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                        nvinfer1::ITensor& input, int c1, int c2, int n, bool a2, int area, bool residual,
                        float mlp_ratio, float e, int g, bool shortcut, std::string lname) {
    int c = (int)(((float)c2) * e);
    int num_heads = c / 32 * 2;

    //assert(c % 32 == 0 && "c2 should be divisible by 32");
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, input, c * 2, {1, 1}, 1, lname + ".cv1");

    if (a2) {

        nvinfer1::ILayer* ablock1 =
                ABlock(network, weightMap, *conv1->getOutput(0), c, num_heads, mlp_ratio, area, lname + ".m.0.0");
        nvinfer1::ILayer* ablock2 =
                ABlock(network, weightMap, *ablock1->getOutput(0), c, num_heads, mlp_ratio, area, lname + ".m.0.1");
        nvinfer1::ILayer* ablock3 =
                ABlock(network, weightMap, *ablock2->getOutput(0), c, num_heads, mlp_ratio, area, lname + ".m.1.0");
        nvinfer1::ILayer* ablock4 =
                ABlock(network, weightMap, *ablock3->getOutput(0), c, num_heads, mlp_ratio, area, lname + ".m.1.1");

        nvinfer1::ITensor* inputTensors[] = {conv1->getOutput(0), ablock2->getOutput(0), ablock4->getOutput(0)};
        nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensors, 3);

        nvinfer1::IElementWiseLayer* conv2 =
                convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1, 1}, 1, lname + ".cv2");
        return conv2;
    } else {

        nvinfer1::ILayer* c3k_ = C3k(network, weightMap, *conv1->getOutput(0), c * 2, c * 2, 2, shortcut, {3, 3},
                                     {3, 3}, 0.5, lname + ".m.0");

        nvinfer1::ITensor* inputTensors[] = {conv1->getOutput(0), c3k_->getOutput(0)};
        nvinfer1::IConcatenationLayer* cat = network->addConcatenation(inputTensors, 2);

        nvinfer1::IElementWiseLayer* conv2 =
                convBnSiLU(network, weightMap, *cat->getOutput(0), c2, {1, 1}, 1, lname + ".cv2");
        return conv2;
    }
}

nvinfer1::ILayer* ABlock(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                         nvinfer1::ITensor& input, int dim, int num_heads, float mlp_ratio, int area,
                         std::string lname) {
    int mlp_hidden_dim = (int)(dim * mlp_ratio);

    nvinfer1::ILayer* attn = AAttn(network, weightMap, input, dim, num_heads, mlp_ratio, area, lname + ".attn");
    nvinfer1::IElementWiseLayer* sum =
            network->addElementWise(input, *attn->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    //mlp
    nvinfer1::IElementWiseLayer* mlp1 =
            convBnSiLU(network, weightMap, *sum->getOutput(0), mlp_hidden_dim * 2, {1, 1}, 1, lname + ".mlp.0");

    nvinfer1::ILayer* mlp2 = convBn(network, weightMap, *mlp1->getOutput(0), dim * 2, 1, 1, lname + ".mlp.1");

    nvinfer1::IElementWiseLayer* sum2 =
            network->addElementWise(*sum->getOutput(0), *mlp2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    return sum2;
}

nvinfer1::ILayer* AAttn(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                        nvinfer1::ITensor& input, int dim, int num_heads, float mlp_ratio, int area,
                        std::string lname) {
    int head_dim = (int)(dim / num_heads);
    int all_head_dim = head_dim * num_heads;
    //TODO: SCALE IS STATIC, CONVERT TO DYNAMIC!
    float scale = 0.176777;
    auto dims = input.getDimensions();
    int B = dims.d[0];
    int C = dims.d[1];
    int H = dims.d[2];
    int W = dims.d[3];
    int N = H * W;

    auto* qkv = convBn(network, weightMap, input, all_head_dim * 3 * 2, 1, 1, lname + ".qkv");

    auto* reshape = network->addShuffle(*qkv->getOutput(0));
    reshape->setReshapeDimensions(nvinfer1::Dims3{B, -1, N});
    reshape->setSecondTranspose(nvinfer1::Permutation{0, 2, 1});

    if (area > 1) {
        B = B * area;
        N = (H * W) / area;
    }

    auto* reshape1 = network->addShuffle(*reshape->getOutput(0));
    reshape1->setReshapeDimensions(nvinfer1::Dims4{B, N, num_heads, head_dim * 3 * 2});
    reshape1->setSecondTranspose(nvinfer1::Permutation{0, 2, 3, 1});

    nvinfer1::ISliceLayer* q = network->addSlice(
            *reshape1->getOutput(0), nvinfer1::Dims4{0, 0, 0, 0},
            nvinfer1::Dims4{reshape1->getOutput(0)->getDimensions().d[0], reshape1->getOutput(0)->getDimensions().d[1],
                            reshape1->getOutput(0)->getDimensions().d[2] / 3,
                            reshape1->getOutput(0)->getDimensions().d[3]},
            nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::ISliceLayer* k = network->addSlice(
            *reshape1->getOutput(0), nvinfer1::Dims4{0, 0, reshape1->getOutput(0)->getDimensions().d[2] / 3, 0},
            nvinfer1::Dims4{reshape1->getOutput(0)->getDimensions().d[0], reshape1->getOutput(0)->getDimensions().d[1],
                            reshape1->getOutput(0)->getDimensions().d[2] / 3,
                            reshape1->getOutput(0)->getDimensions().d[3]},
            nvinfer1::Dims4{1, 1, 1, 1});
    nvinfer1::ISliceLayer* v = network->addSlice(
            *reshape1->getOutput(0), nvinfer1::Dims4{0, 0, 2 * reshape1->getOutput(0)->getDimensions().d[2] / 3, 0},
            nvinfer1::Dims4{reshape1->getOutput(0)->getDimensions().d[0], reshape1->getOutput(0)->getDimensions().d[1],
                            reshape1->getOutput(0)->getDimensions().d[2] / 3,
                            reshape1->getOutput(0)->getDimensions().d[3]},
            nvinfer1::Dims4{1, 1, 1, 1});

    auto* qT = network->addShuffle(*q->getOutput(0));
    qT->setFirstTranspose(nvinfer1::Permutation{0, 1, 3, 2});

    auto matmul = network->addMatrixMultiply(*qT->getOutput(0), nvinfer1::MatrixOperation::kNONE, *k->getOutput(0),
                                             nvinfer1::MatrixOperation::kNONE);

    float* scale_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    scale_val[0] = scale;
    nvinfer1::Weights s_w{nvinfer1::DataType::kFLOAT, scale_val, 1};
    float* shift_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    shift_val[0] = 0;
    nvinfer1::Weights sh_w{nvinfer1::DataType::kFLOAT, shift_val, 1};
    float* power_val = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    power_val[0] = 1;
    nvinfer1::Weights p_w{nvinfer1::DataType::kFLOAT, power_val, 1};
    nvinfer1::IScaleLayer* mul =
            network->addScale(*matmul->getOutput(0), nvinfer1::ScaleMode::kUNIFORM, sh_w, s_w, p_w);
    auto* softmax = network->addSoftMax(*mul->getOutput(0));
    softmax->setAxes(1 << 3);

    auto transpose3 = network->addShuffle(*softmax->getOutput(0));
    transpose3->setFirstTranspose(nvinfer1::Permutation{0, 1, 3, 2});

    auto matmul1 = network->addMatrixMultiply(*v->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                              *transpose3->getOutput(0), nvinfer1::MatrixOperation::kNONE);

    auto transpose4 = network->addShuffle(*matmul1->getOutput(0));
    transpose4->setFirstTranspose(nvinfer1::Permutation{0, 3, 1, 2});

    if (area > 1) {
        B = B / area;
        N = N * area;
    }

    auto* reshape3 = network->addShuffle(*transpose4->getOutput(0));
    reshape3->setReshapeDimensions(nvinfer1::Dims4{B, H, W, -1});

    auto* transpose6 = network->addShuffle(*reshape3->getOutput(0));
    transpose6->setFirstTranspose(nvinfer1::Permutation{0, 3, 1, 2});

    auto transpose5 = network->addShuffle(*v->getOutput(0));
    transpose5->setFirstTranspose(nvinfer1::Permutation{0, 3, 1, 2});

    auto* reshape4 = network->addShuffle(*transpose5->getOutput(0));
    reshape4->setReshapeDimensions(nvinfer1::Dims4{B, H, W, C});

    //reshape4->setSecondTranspose(nvinfer1::Permutation{0, 3, 1, 2});
    auto* transpose7 = network->addShuffle(*reshape4->getOutput(0));
    transpose7->setFirstTranspose(nvinfer1::Permutation{0, 3, 1, 2});
    auto* pe = convBn(network, weightMap, *transpose7->getOutput(0), all_head_dim * 2, 7, 1, lname + ".pe",
                      all_head_dim * 2);

    auto* sum =
            network->addElementWise(*pe->getOutput(0), *transpose6->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    auto* proj = convBn(network, weightMap, *sum->getOutput(0), all_head_dim * 2, 1, 1, lname + ".proj");

    return proj;
}

这是config.h里的内容：
#define USE_FP16
// #define USE_FP32
// #define USE_INT8

const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "output";
const static char* kProtoTensorName = "proto";
const static int kNumClass = 3;
const static int kPoseNumClass = 1;
const static int kNumberOfPoints = 17;  // number of keypoints total
// obb model's number of classes
constexpr static int kObbNumClass = 15;
const static int kObbNe = 1;  // number of extra parameters
const static int kBatchSize = 1;
const static int kGpuId = 0;
const static int kInputH = 640;
const static int kInputW = 640;
const static int kObbInputH = 1024;
const static int kObbInputW = 1024;
const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;
const static float kConfThreshKeypoints = 0.5f;  // keypoints confidence
const static int kMaxInputImageSize = 3000 * 3000;
const static int kMaxNumOutputBbox = 1000;
//Quantization input image folder path
const static char* kInputQuantizationFolder = "./coco_calib";

// Classfication model's number of classes
constexpr static int kClsNumClass = 3;
// Classfication model's input shape
constexpr static int kClsInputH = 224;
constexpr static int kClsInputW = 224;
