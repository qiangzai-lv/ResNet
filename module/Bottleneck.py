import torch.nn as nn


class Bottleneck(nn.Module):
    """
    Bottleneck模块用于ResNet50和ResNet101基本残差结构块，由三个3x3卷积层和一个1x1卷积层组成。
    """
    def __init__(self, inChannel, outChannel, stride=1):
        super(Bottleneck, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inChannel, int(outChannel / 4), kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(int(outChannel / 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outChannel / 4), int(outChannel / 4), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(outChannel / 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outChannel / 4), outChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outChannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inChannel != outChannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outChannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = nn.ReLU(inplace=True)(out)
        return out
