import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, channel, initial=True):
        super(Generator, self).__init__()
        self.init_size = image_size
        self.l3 = nn.Linear(latent_dim, 128 * image_size ** 2)
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 3, 3, stride = 1, padding = 1),
        )
    
        if initial:
            self.init()

    def init(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)

    def forward(self, z):
        out1 = self.l3(z)
        out = out1.view(out1.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_block(out)
        return img 


class Feature_Decoder(nn.Module):
    def __init__(self):
        super(Feature_Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv1 = nn.Conv2d(512, 256, 1, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(256, 128, 1, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(128, 64, 1, stride = 1, padding = 0)
        self.conv4 = nn.Conv2d(64, 3, 1, stride = 1, padding = 0)
        self.conv5 = nn.Conv2d(64, 3, 1, stride = 1, padding = 0)
        self.conv6 = nn.Conv2d(3, 3, 1, stride = 1, padding = 0)
        self.conv_31 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_32 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_33 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_34 = nn.Conv2d(3, 3, 3, stride=1, padding=1)

    def forward(self, x, f1, f2, f3, f4, f5):
        out = self.conv1(self.upsample(f5))
        out = self.conv_31(out + f4)
        
        out = self.conv2(self.upsample(out))
        out = self.conv_32(out + f3)
        
        out = self.conv3(self.upsample(out))
        out = self.conv_33(out + f2)
        
        out_ = self.conv4(out)
        out = (x + out_)
        out = self.conv6(out)
        out = torch.nn.functional.tanh(out)
        
        return out, out_


