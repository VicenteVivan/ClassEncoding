from binascii import a2b_hex
from unicodedata import name
from matplotlib.pyplot import get
import torch
import torch.nn as nn
import numpy as np
import math

from transformers import ViTModel
from transformers import ResNetForImageClassification as ResNet

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = ResNet.from_pretrained('microsoft/resnet-50', output_hidden_states=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten(1,-1)
        self.mlp = nn.Linear(2048, 100)
    
    def forward(self, image):
        out = self.resnet(image).hidden_states[-1]
        out = self.flatten(self.avgpool(out))
        out = self.mlp(out)
        return out

class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = ResNet.from_pretrained('microsoft/resnet-50', output_hidden_states=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten(1,-1)

        self.mlp = nn.Linear(2048, 512)
    
    def forward(self, image):
        out = self.resnet(image).hidden_states[-1]
        out = self.flatten(self.avgpool(out))
        out = self.mlp(out)
        return out

if __name__ == "__main__":
    model = ResNet50()
    
    # Test Model 1
    X = torch.rand((1,3,224,224))
    Y = model(X)
    print(Y.shape)
    

