import torch.nn as nn

from module.BasicBlock import BasicBlock
import torch.nn.functional as F

from module.Bottleneck import Bottleneck

'''-------------ResNet18---------------'''


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.inChannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inChannel, channels, stride))
            self.inChannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):  # 3*32*32
        out = self.conv1(x)  # 64*32*32
        out = self.layer1(out)  # 64*32*32
        out = self.layer2(out)  # 128*16*16
        out = self.layer3(out)  # 256*8*8
        out = self.layer4(out)  # 512*4*4
        out = F.avg_pool2d(out, 4)  # 512*1*1
        out = out.view(out.size(0), -1)  # 512
        out = self.fc(out)
        return out


'''-------------ResNet34---------------'''


class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        self.inChannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self.make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, 3, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inChannel, channels, stride))
            self.inChannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):  # 3*32*32
        out = self.conv1(x)  # 64*32*32
        out = self.layer1(out)  # 64*32*32
        out = self.layer2(out)  # 128*16*16
        out = self.layer3(out)  # 256*8*8
        out = self.layer4(out)  # 512*4*4
        out = F.avg_pool2d(out, 4)  # 512*1*1
        out = out.view(out.size(0), -1)  # 512
        out = self.fc(out)
        return out


'''-------------ResNet50---------------'''


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.inChannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(Bottleneck, 256, 3, stride=1)
        self.layer2 = self.make_layer(Bottleneck, 512, 4, stride=2)
        self.layer3 = self.make_layer(Bottleneck, 1024, 6, stride=2)
        self.layer4 = self.make_layer(Bottleneck, 2048, 3, stride=2)
        self.fc = nn.Linear(512 * 4, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inChannel, channels, stride))
            self.inChannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):  # 3*32*32
        out = self.conv1(x)  # 64*32*32
        out = self.layer1(out)  # 64*32*32
        out = self.layer2(out)  # 128*16*16
        out = self.layer3(out)  # 256*8*8
        out = self.layer4(out)  # 512*4*4
        out = F.avg_pool2d(out, 4)  # 512*1*1
        out = out.view(out.size(0), -1)  # 512
        out = self.fc(out)
        return out
