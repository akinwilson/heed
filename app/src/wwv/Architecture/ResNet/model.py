import torch.nn as nn 
import torch.nn.functional as F 
from .layers import Bottleneck


class ResNet(nn.Module):
    def __init__(self, num_blocks, cfg, block=Bottleneck, num_classes=1, dropout=0.2 ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.cfg= cfg 
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.dropout = nn.Dropout(p=dropout)
        # need to parameterised. Given input size, I know what the linear's layers row dim should be
        self.linear = nn.Linear(20480, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        out = F.relu(self.bn1(x))
        # print("outF.relu(self.bn1(x))", out.shape)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        # equivalent to flatten operation
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out
