"""
参数冻结
"""
import torch
from torchvision.models import resnet18

model = resnet18(pretrained=True)
trainable_layers = [] #自定义修改

for name, param in model.named_parameters():
    name_layer = name.split('.')[0]
    if name_layer in trainable_layers:
        param.requires_grad = True
    else:
        param.requires_grad = True

"""
要重新训练的层初始化权重
"""
def reinitialize_weights(layer_weight):
    torch.nn.init.xavier_uniform_(layer_weight)

# reinitialize trainable layers
for layer_name, layer in model.named_children():
    if isinstance(layer, torch.nn.Linear) and layer_name in trainable_layers:
        reinitialize_weights(layer.weight)
        layer.bias.data.fill_(0.01)