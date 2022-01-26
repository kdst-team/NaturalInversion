'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

vgg11_layer = [ 
                '0_Conv2d',
                '1_BatchNorm2d',
                '2_ReLU',
                '3_MaxPool2d',
                '4_Conv2d',
                '5_BatchNorm2d',
                '6_ReLU',
                '7_MaxPool2d',
                '8_Conv2d', 
                '9_BatchNorm2d', 
                '10_ReLU', 
                '11_Conv2d', 
                '12_BatchNorm2d', 
                '13_ReLU', 
                '14_MaxPool2d', 
                '15_Conv2d', 
                '16_BatchNorm2d', 
                '17_ReLU', 
                '18_Conv2d', 
                '19_BatchNorm2d', 
                '20_ReLU', 
                '21_MaxPool2d', 
                '22_Conv2d', 
                '23_BatchNorm2d', 
                '24_ReLU', 
                '25_Conv2d', 
                '26_BatchNorm2d', 
                '27_ReLU', 
                '28_MaxPool2d'
            ]   

vgg16_layer = [
        '0_Conv2d',
        '1_BatchNorm2d',
        '2_ReLU',
        '3_Conv2d',
        '4_BatchNorm2d',
        '5_ReLU',
        '6_MaxPool2d',
        '7_Conv2d',
        '8_BatchNorm2d',
        '9_ReLU',
        '10_Conv2d',
        '11_BatchNorm2d',
        '12_ReLU',
        '13_MaxPool2d',
        '14_Conv2d',
        '15_BatchNorm2d',
        '16_ReLU',
        '17_Conv2d',
        '18_BatchNorm2d',
        '19_ReLU',
        '20_Conv2d',
        '21_BatchNorm2d',
        '22_ReLU',
        '23_MaxPool2d',
        '24_Conv2d',
        '25_BatchNorm2d',
        '26_ReLU',
        '27_Conv2d',
        '28_BatchNorm2d',
        '29_ReLU',
        '30_Conv2d',
        '31_BatchNorm2d',
        '32_ReLU',
        '33_MaxPool2d',
        '34_Conv2d',
        '35_BatchNorm2d',
        '36_ReLU',
        '37_Conv2d',
        '38_BatchNorm2d',
        '39_ReLU',
        '40_Conv2d',
        '41_BatchNorm2d',
        '42_ReLU',
        '43_MaxPool2d'
    ]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x, arch='vgg11'):
        x_dict = self.features(x, flatten=True)
        #for key in x_dict:
        #    print(key)
        last_key = None
        for key in x_dict:
            last_key = key
        '''
        if arch == 'vgg11':
            x = x_dict[vgg11_layer[-1]]
        elif arch == 'vgg16':
            x = x_dict[vgg16_layer[-1]]
        '''
        x = x_dict[last_key]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, x_dict


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn(num_classes=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes)


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn(num_classes=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes)


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
