import torch
import torch.nn as nn
from stream.scnn import StreamingCNN, StreamingConv2d
from torchvision.models import resnet18, resnet34, resnet50

#%%
resnet = resnet18(weights="IMAGENET1K_V1")

def split_model(model):
    stream_net = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    head = nn.Sequential(model.avgpool, nn.Flatten(), model.fc)
    return stream_net, head

stream_net, head = split_model(resnet)

print(type(head))
