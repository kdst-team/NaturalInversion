# 2019.07.24-Changed output of forward function
# Huawei Technologies Co., Ltd. <foss@huawei.com>
# taken from https://github.com/huawei-noah/Data-Efficient-Model-Compression/blob/master/DAFL/resnet.py
# for comparison with DAFL


import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1,p=0.0):
        super(BasicBlock, self).__init__()
        self.p = p
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            out += self.shortcut(x)
        out = self.relu2(out)
        output_do = F.dropout(out,p=self.p,training=True)
        return output_do
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1,p=0.0):
        super(Bottleneck, self).__init__()
        self.p = p
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        #output_do = F.dropout(out,p=self.p,trainig=True)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,p=0.0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.p=p 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, out_feature=False):
        x = self.conv1(x)

        x = self.bn1(x)
        out0 = F.relu(x)
        #out = F.dropout(out,p=self.p,training=True)

        out = self.layer1(out0)
        out1 = self.layer2(out)
        out2 = self.layer3(out1)
        out3 = self.layer4(out2)
        #out = F.avg_pool2d(out, 4)
        out4 = self.avgpool(out3)
        #out = F.dropout(out,p=self.p,training=True)
        '''
        print("layer 1 ",out.shape)
        print("layer 2 ",out1.shape)
        print("layer 3 : ",out2.shape)
        print("layer 4 : ",out3.shape)
        print("avgpool : ",out4.shape)
        '''
        feature = out4.view(out4.size(0), -1)
        img = self.linear(feature)
        if out_feature == False:
            return img, out3, out2, out1, out, out0
        else:
            return img, feature, out3, out2, out1, out, out0
 
 
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)
 
def ResNet34(num_classes=10,p=0.0):
    return ResNet(BasicBlock, [3,4,6,3], num_classes,p)
 
def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)
 
def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)
 
def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)
 
