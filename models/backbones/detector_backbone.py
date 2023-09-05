import ipdb
import torch
import torch.nn as nn
from MGCA.models.backbones import cnn_backbones
from torch import nn


class ResNetDetector(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()

        model_function = getattr(cnn_backbones, model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(  # ?  # resnet，resnet输出特征维度，1024
            pretrained=pretrained
        )

        if model_name == "resnet_50":
            self.filters = [512, 1024, 2048]  # 三种feature map的通道数

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # resnet第一个卷积block
        out3 = self.model.layer2(x)   # bz, 512, 28  # resnet第2个卷积block
        out4 = self.model.layer3(out3)  # resnet第3个卷积block
        out5 = self.model.layer4(out4)  # resnet第4个卷积block

        return out3, out4, out5  # 第二个block的输出，第三个block的输出，第四个block的输出


if __name__ == "__main__":
    model = ResNetDetector("resnet_50")
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    ipdb.set_trace()
