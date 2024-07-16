import torch.nn as nn


class BasicBlock(nn.Module):
    """
    BasicBlock模块用于ResNet18和ResNet34基本残差结构块，由两个3x3卷积层和一个1x1卷积层组成。
    """

    def __init__(self, inChannel, outChannel, stride=1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),  # inplace=True表示进行原地操作，一般默认为False，表示新建一个变量存储操作
            nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outChannel)
        )
        self.shoutCut = nn.Sequential()
        if stride != 1 or inChannel != outChannel:
            self.shoutCut = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outChannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shoutCut(x)
        out = nn.ReLU(inplace=True)(out)
        return out
