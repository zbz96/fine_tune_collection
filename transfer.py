"""
迁移学习调用模型的方法：
    1. 使用list列表直接截取
    2. 直接对结构进行修改
"""

import torch
import  torch.nn as nn
from torchvision.models import resnet18

train_model = resnet18(pretrained=True)

"""
在pytorch中children() 和 modules() 的区别：
    modules() 迭代模型的所有子层
    children() 只迭代模型的第一层子层

model_children = [x for x in train_model.children()] 
model_modules = [x for x in train_model.modules()]
print(len(model_children))
print(len(model_modules))

"""
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    
    def forward(self,x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1,shape)


# 1. 使用list列表直接截取
model = nn.Sequential(
        *list(train_model.children())[:-1], # torch.Size([B, 512, 1, 1])
        Flatten(),# torch.Size([B, 512])
        nn.Linear(512,5))

# 2. 直接对结构进行修改
finetune_net = resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
"""
for layer in model.modules():
    if isinstance(layer, nn.Conv2d):
        ...对layer进行处理

for name,layer in model.name_modules():
     if 'conv' in name:
        ...对layer进行处理
"""




