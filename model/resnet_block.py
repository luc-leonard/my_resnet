from fastai.layers import ConvLayer, BatchNorm, NormType
from torch import nn
import torch.nn.functional as F


def noop(x):
    return x


def zero_norm(nf):
    norm = nn.BatchNorm2d(nf)
    norm.bias.data.fill_(1e-3)
    norm.weight.data.fill_(0.)
    return norm


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self._conv = nn.Sequential(
            ConvLayer(ni, nf // 4, 1),
            ConvLayer(nf // 4, nf // 4, stride=stride),
            ConvLayer(nf // 4, nf, 1, act_cls=None, norm_type=NormType.BatchZero)
        )
        # self._conv = nn.Sequential(
        #     nn.Conv2d(ni, nf // 4, kernel_size=(1, 1)),
        #     nn.ReLU(),
        #     nn.Conv2d(nf // 4, nf // 4, kernel_size=(3, 3), stride=stride, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(nf // 4, nf, kernel_size=(1, 1)),
        #     zero_norm(nf)
        # )

        self.idconv = noop if ni == nf else \
            nn.Sequential(
                nn.Conv2d(ni, nf, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(nf)

            )
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu_(self._conv(x) + self.idconv(self.pool(x)))