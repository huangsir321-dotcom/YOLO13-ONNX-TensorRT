# YOLOv13-ONNX-TensorRT

> YOLOv13 从训练到模型部署全实战

本项目是一个完整的 **YOLOv13 目标检测** 落地实战项目，覆盖了从 **数据准备 → 模型训练 → 模型导出 → ONNX Runtime 部署 → TensorRT 部署（Python / C++）→ 摄像头实时检测** 的完整链路，最终实现在嵌入式/服务器 GPU 上的高效推理。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 项目做了什么（全流程概览）](#2-项目做了什么全流程概览)
- [3. 目录结构](#3-目录结构)
- [4. 环境依赖](#4-环境依赖)
- [5. YOLOv13 算法原理](#5-yolov13-算法原理)
- [6. 数据准备](#6-数据准备)
- [7. 模型训练](#7-模型训练)
- [8. 模型导出](#8-模型导出)
- [9. ONNX Runtime 部署](#9-onnx-runtime-部署)
- [10. TensorRT 部署](#10-tensorrt-部署)
- [11. 各推理脚本说明](#11-各推理脚本说明)
- [12. 常见问题](#12-常见问题)
- [13. 参考与致谢](#13-参考与致谢)

---

## 1. 项目简介

本项目以 **YOLOv13**（基于 ultralytics 框架）为目标检测算法，实现了猫（cat）、杯子（cup）、狗（dog）三类目标的检测任务，并完成以下工作：

1. 使用 `labelImg` 完成数据标注；
2. 编写数据划分脚本，将数据集划分为训练集 / 验证集 / 测试集；
3. 基于 ultralytics 完成 YOLOv13 模型训练；
4. 将训练好的 `.pt` 模型导出为 **ONNX** 格式；
5. 分别使用 **ONNX Runtime**（Python / C++）完成推理部署；
6. 使用 **TensorRT**（Python / C++）完成高性能推理部署，并支持 FP32 / FP16 / INT8 精度；
7. 实现基于摄像头的实时目标检测（含多线程双缓冲优化，可达高帧率）。

**检测类别（3 类）：**

| ID | 类别 |
|----|------|
| 0  | cat  |
| 1  | cup  |
| 2  | dog  |

---

## 2. 项目做了什么（全流程概览）

整个项目的完整流程如下图所示：

```
原始图片 + 标注(labelImg)
        │
        ▼
数据划分 (huafen.py)  ── 训练集 75% / 验证集 15% / 测试集 15%
        │
        ▼
YOLOv13 训练 (ultralytics, dataset.yaml)
        │
        ▼
best.pt（PyTorch 权重）
        │
        ├──────────────────────────┐
        │                          │
        ▼                          ▼
模型导出 export.py           模型导出 gen_wts.py
  (.pt → .onnx, FP16)          (.pt → .wts)
        │                          │
        ▼                          ▼
ONNX Runtime 部署          TensorRT 部署 (trtexec)
  · Python                    · .onnx → .engine (FP16)
  · C++                       · .wts  → .engine (C++ API)
        │                          │
        ▼                          ▼
   图片 / 摄像头实时检测        图片 / 摄像头实时检测
```

**具体做了哪些事情：**

- ✅ 环境搭建：CUDA、cuDNN、TensorRT、OpenCV、ONNX Runtime 的安装配置；
- ✅ 数据标注与划分：`huafen.py` 按比例切分数据集；
- ✅ 模型训练：YOLOv13n 模型在自定义数据集上训练；
- ✅ PyTorch 推理：`predict.py`、`cam.py`；
- ✅ ONNX 导出与推理：Python 端 `onnxpredict.py` 等，C++ 端 `yolov13_onnx_cpp`；
- ✅ TensorRT 引擎构建与推理：Python 端 `yolo13_det_trt*.py`，C++ 端 `tensorrtx-yolov13`；
- ✅ 摄像头实时检测（多线程双缓冲、高帧率优化）。

---

## 3. 目录结构

```
YOLOv13-ONNX-TensorRT/
├── 操作教程                    # 完整的环境搭建与部署操作步骤文档
├── dataset.yaml                # 数据集配置文件（3 类）
├── yolov13n.pt                 # YOLOv13n 预训练权重
├── yolov13.engine              # TensorRT 引擎文件
├── test_cup.jpg                # 测试图片
│
├── coco128/                    # COCO128 示例数据集
├── yolov13-main/               # YOLOv13 训练主代码（ultralytics）
│   ├── huafen.py               # 数据集划分脚本
│   ├── export.py               # ONNX 导出脚本
│   ├── gen_wts.py              # .pt → .wts 权重转换脚本
│   ├── predict.py              # PyTorch 图片推理
│   ├── cam.py / cam1.py / cam2.py   # PyTorch 摄像头推理
│   ├── onnxpredict.py          # ONNX Runtime Python 推理
│   ├── onnxpre.py / onnxdect.py     # ONNX 预处理/检测脚本
│   ├── best.pt / best.onnx / best.engine  # 训练产物
│   └── runs/                   # 训练结果（权重、曲线、混淆矩阵等）
│
├── yolov13_onnx_cpp/           # ONNX Runtime C++ 部署工程
│   ├── src/main.cpp            # 图片推理入口
│   ├── src/cam.cpp             # 摄像头推理入口
│   ├── src/preprocess.cpp      # 预处理
│   ├── src/postprocess.cpp     # 后处理
│   ├── src/utils.cpp           # 工具函数
│   └── CMakeLists.txt
│
├── tensorrtx-yolov13/          # TensorRT C++ / Python 部署工程
│   ├── yolo13_det_trt.py       # TensorRT Python 图片推理
│   ├── yolo13_det_trt_cam.py   # TensorRT Python 摄像头推理
│   ├── yolo13_det_trt_cam2.py  # TensorRT Python 摄像头推理（改进版）
│   ├── gen_wts.py              # .pt → .wts 转换
│   ├── yolov13enginecam.cpp    # TensorRT C++ 摄像头推理（多线程双缓冲）
│   ├── yolov13enginecam120fps.cpp  # 高帧率版本
│   ├── plugin/yololayer.cu     # YOLO 层 CUDA 插件
│   ├── src/                    # 前后处理、模型加载等
│   └── CMakeLists.txt
│
├── tensorrt/                   # tensorrtx 参考实现（C++ 图片推理）
├── onnxruntime/                # ONNX Runtime 相关（含 yolov13n.onnx）
└── onnxruntime-linux-x64-1.18.1/   # ONNX Runtime 1.18.1 运行库
```

---

## 4. 环境依赖

| 依赖 | 版本 |
|------|------|
| 操作系统 | Ubuntu 20.04 / 22.04（Linux） |
| CUDA | 11.8（也可 11.6） |
| cuDNN | 8.9.1.23 |
| TensorRT | 8.6.1.6 |
| OpenCV | 3.4.3 / 4.8.0 |
| ONNX Runtime | 1.18.1 |
| PyTorch | 2.2.2+cu118 |
| torchvision | 对应版本 |
| ultralytics | 8.3.63 |
| Python | 3.11 |

详细的安装步骤见 [操作教程](操作教程) 文档。

---

## 5. YOLOv13 算法原理

### 5.1 目标检测的基本思想

目标检测要解决两个问题：**物体在哪里（定位）** 和 **物体是什么（分类）**。YOLO 系列属于 **一阶段（one-stage）** 检测器，其核心思想是：

> 将整张图片一次性输入网络，直接在输出端同时回归出所有目标框的 **位置坐标 + 类别置信度**，不需要候选框提取（region proposal）阶段，因此速度快、适合实时场景。

### 5.2 网络结构

YOLOv13 沿用 ultralytics 框架的经典三段式结构，由三部分组成：

1. **Backbone（骨干网络）**
   负责提取图像的多尺度特征。通过一系列卷积 + CSP（跨阶段局部连接）结构，逐层下采样（如 640→320→160→80→40→20），得到不同分辨率的特征图，捕捉从低级纹理到高级语义的信息。

2. **Neck（颈部网络）**
   采用 **FPN + PAN** 结构进行多尺度特征融合：
   - **FPN（特征金字塔）**：自顶向下，把高层的语义信息上采样后传递给低层；
   - **PAN（路径聚合）**：自底向上，把低层的定位信息传递回高层。
   
   两者结合，让不同尺度的特征图既包含丰富的语义，又保留精确的位置信息，提升对不同大小目标的检测能力。

3. **Head（检测头）**
   采用 **解耦头（decoupled head）+ 无锚框（anchor-free）** 设计：
   - 不再预定义 anchor 框，直接预测每个网格点的目标中心；
   - 将分类分支与回归分支分开，分别输出类别概率和边界框信息；
   - 在 **3 个尺度**（80×80、40×40、20×20）上分别预测，对应检测小、中、大目标。

### 5.3 输出张量解析

以本项目 640×640 输入、3 个类别为例，网络输出形状为：

```
(4 + num_classes, 8400) = (7, 8400)
```

其中：

| 维度 | 含义 |
|------|------|
| 第 0~3 行 | 边界框中心点坐标 (cx, cy) 和宽高 (w, h) |
| 第 4~6 行 | 3 个类别的置信度分数 |
| 8400 | 候选框总数 = 80×80 + 40×40 + 20×20 = 6400 + 1600 + 400 |

### 5.4 训练损失

YOLOv13 的训练损失主要包括：

- **分类损失（Classification Loss）**：BCE 损失，衡量类别预测是否准确；
- **回归损失（Regression Loss）**：使用 CIoU 等损失，衡量预测框与真实框的重合程度；
- **置信度损失（Objectness Loss）**：判断该位置是否存在目标。

### 5.5 推理流程（预处理 → 前向 → 后处理）

1. **预处理（Preprocess）**
   - 读取图片，颜色空间 BGR → RGB；
   - **Letterbox 等比缩放**：保持宽高比缩放到 640×640，不足部分用灰色（114, 114, 114）填充；
   - 归一化到 [0, 1]（除以 255）；
   - 通道顺序 HWC → CHW，并增加 batch 维度，得到 `(1, 3, 640, 640)`。

2. **前向推理（Forward）**
   输入网络，得到原始输出张量 `(7, 8400)`。

3. **后处理（Postprocess）**
   - 取出 8400 个候选框，对每个框取 3 类中的 **最大置信度** 及其类别 ID；
   - **置信度过滤**：丢弃低于阈值（本项目 `conf_thresh = 0.5`）的框；
   - **坐标还原**：将预测坐标从 letterbox 空间映射回原图尺寸（去除 padding、除以缩放比例）；
   - **非极大值抑制（NMS）**：对同一物体产生的多个重叠框，按 IoU 阈值（`iou_threshold = 0.4`）去重，保留最优框；
   - 绘制检测框与标签。

---

## 6. 数据准备

### 6.1 数据标注

使用 `labelImg` 对原始图片进行标注，生成 YOLO 格式的 `.txt` 标签文件（每行：`class_id cx cy w h`，坐标为归一化值）。

### 6.2 数据划分

`huafen.py` 将数据集按比例划分为：

- 训练集：75%
- 验证集：15%
- 测试集：15%

脚本会自动创建对应目录并复制图片与标签。

### 6.3 数据集配置

`dataset.yaml` 指定了数据集路径与类别：

```yaml
train: .../images/huafen/images/train
val:   .../images/huafen/images/val

names:
  0: cat
  1: cup
  2: dog
```

---

## 7. 模型训练

使用 ultralytics 框架训练 YOLOv13：

```python
from ultralytics import YOLO

model = YOLO('yolov13n.pt')          # 或 yolov13s / l / x
model.train(data='dataset.yaml',
            epochs=...,              # 训练轮数
            imgsz=640,               # 输入尺寸
            batch=...,               # batch 大小
            device=0)                # GPU
```

训练完成后，在 `runs/train/exp*/weights/` 下得到 `best.pt`（最优权重）和 `last.pt`（最后权重），同时生成训练曲线、PR 曲线、混淆矩阵等可视化结果。

---

## 8. 模型导出

### 8.1 导出 ONNX

`export.py` 将 PyTorch 权重导出为 FP16 的 ONNX 模型：

```python
from ultralytics import YOLO

model = YOLO('runs/train/exp3/weights/best.pt')
model.export(format="onnx", half=True)   # FP16
```

### 8.2 导出 WTS（用于 TensorRT C++）

`gen_wts.py` 将 `.pt` 权重转换为 TensorRT C++ 工程所需的 `.wts` 文本格式：

```shell
python gen_wts.py -w yolov13n.pt -t detect -o yolov13n.wts
```

### 8.3 生成 TensorRT 引擎（engine）

方式一：使用 `trtexec` 工具（命令行）：

```shell
trtexec --onnx=best.onnx --saveEngine=yolov13.engine --fp16
```

方式二：使用 C++ API 从 `.wts` 构建引擎（见 `tensorrtx-yolov13`）。

---

## 9. ONNX Runtime 部署

### 9.1 Python 推理

- `onnxpredict.py`：使用 ONNX Runtime 加载 ONNX 模型，完成预处理 → 推理 → 后处理 → 结果可视化。

### 9.2 C++ 推理

`yolov13_onnx_cpp` 工程使用 ONNX Runtime C++ API 完成部署：

```shell
cd yolov13_onnx_cpp
mkdir build && cd build
cmake ..
make
./yolov13_image   # 图片检测
./yolov13_cam     # 摄像头检测
```

---

## 10. TensorRT 部署

TensorRT 是 NVIDIA 的高性能深度学习推理优化器，通过 **算子融合、精度校准（FP16/INT8）、内核自动调优** 等方式大幅提升推理速度，适合在 NVIDIA GPU 上部署。

### 10.1 Python 推理

| 脚本 | 功能 |
|------|------|
| `yolo13_det_trt.py` | 图片推理 |
| `yolo13_det_trt_cam.py` | 摄像头实时推理 |
| `yolo13_det_trt_cam2.py` | 摄像头实时推理（改进版，支持动态类别） |

核心流程：加载 `.engine` → 分配显存 → 预处理 → `execute_async_v3` 推理 → 后处理 → 绘制结果。

### 10.2 C++ 推理

`tensorrtx-yolov13` 工程使用 TensorRT C++ API + CUDA 插件实现：

```shell
cd tensorrtx-yolov13
mkdir build && cd build
cmake ..
make
./yolov13_camera
```

该工程特点：

- 使用自定义 **YOLO 层 CUDA 插件**（`plugin/yololayer.cu`）完成高效后处理；
- `yolov13enginecam.cpp` 采用 **多线程 + 双缓冲队列**，将图像采集、预处理、推理解耦，提升实时性；
- `yolov13enginecam120fps.cpp` 为高帧率优化版本；
- 支持 FP32 / FP16 / INT8（INT8 需准备校准集并开启 `USE_INT8` 宏）。

---

## 11. 各推理脚本说明

| 脚本 | 说明 |
|------|------|
| `predict.py` | PyTorch 图片推理 |
| `cam.py` / `cam1.py` / `cam2.py` | PyTorch 摄像头实时检测 |
| `onnxpredict.py` / `onnxpre.py` / `onnxdect.py` | ONNX Runtime Python 推理 |
| `yolov13_onnx_cpp` | ONNX Runtime C++ 图片/摄像头推理 |
| `yolo13_det_trt.py` | TensorRT Python 图片推理 |
| `yolo13_det_trt_cam.py` / `cam2.py` | TensorRT Python 摄像头推理 |
| `yolov13enginecam.cpp` | TensorRT C++ 摄像头推理（多线程双缓冲） |

---

## 12. 常见问题

1. **labelImg 无法启动（Qt 插件报错）**
   ```shell
   sudo apt install libxcb-*
   ```

2. **pycuda 报错 `cuInit failed: initialization error`**
   ```shell
   sudo apt-get install nvidia-modprobe
   # 安装后务必重启系统
   ```

3. **`trtexec` 命令找不到**
   将 TensorRT 的 bin 目录加入 PATH：
   ```shell
   export PATH=/path/to/TensorRT-8.6.1.6/bin:$PATH
   source ~/.bashrc
   ```

4. **GitHub 上传大文件警告**
   `best.engine`（约 68MB）超过 GitHub 建议的 50MB 限制，可考虑使用 Git LFS 或加入 `.gitignore`。

---

## 13. 参考与致谢

- [ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv13 训练框架
- [tensorrtx](https://github.com/wang-xinyu/tensorrtx) — TensorRT C++ 部署参考实现
- [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) — YOLOv13 模型
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — 跨平台推理引擎
- [TensorRT](https://developer.nvidia.com/tensorrt) — NVIDIA 高性能推理优化器