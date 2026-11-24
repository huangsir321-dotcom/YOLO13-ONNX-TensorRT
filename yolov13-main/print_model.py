# import torch
# model = torch.load("/home/bpz/yolo系列网课/yolov13-main/best.pt")
# print(model.model[0])  # 查看第一个卷积层的 out_channels

# import torch

# # 加载文件，得到一个字典
# checkpoint = torch.load('/home/bpz/yolo系列网课/yolov13-main/best.pt') # 请替换为您的实际路径

# # 方法一：如果字典的 'model' 键对应的是模型的状态字典 (state_dict)
# # 这种情况最常见。您需要先创建模型架构，然后加载状态。
# # 但这需要您有定义模型架构的代码。

# # 方法二：如果字典的 'model' 键直接对应模型对象本身（不太常见，但有可能）
# # 这是解决您当前错误最快的方法。
# if 'model' in checkpoint:
#     model = checkpoint['model'] # 从字典中取出 'model' 键对应的值
#     print(model.model[0])       # 现在 model 是模型对象，可以访问 .model 属性
# else:
#     # 如果没有 'model' 键，打印整个字典的键，看看里面有什么
#     print("Loaded checkpoint keys:", checkpoint.keys())
#     # 然后根据实际的键名来访问，例如：
#     # if 'state_dict' in checkpoint:
#     #     print("Keys in state_dict:", checkpoint['state_dict'].keys())

import torch

checkpoint = torch.load('/home/bpz/yolo系列网课/yolov13-main/best.pt')
print("Checkpoint keys:", checkpoint.keys())

# 如果有 'model' 键，打印它的类型
if 'model' in checkpoint:
    print("Type of checkpoint['model']:", type(checkpoint['model']))
    # 如果它是 nn.Module，就可以直接用
    if hasattr(checkpoint['model'], 'model'): # 假设模型对象有一个 .model 属性
        print(checkpoint['model'].model[0].conv.weight.shape)