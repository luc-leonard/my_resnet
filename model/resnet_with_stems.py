from fastai.layers import ConvLayer, Flatten
from torch import nn

from .resnet_block import ResBlock


def resnet_stem(*sizes):
    conv_layers = [ConvLayer(sizes[i], sizes[i + 1], ks=3, stride=2 if i == 0 else 1)
                   for i in range(len(sizes) - 1)]
    return nn.Sequential(*conv_layers, nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class Resnet(nn.Sequential):
    def __init__(self, out_classes, layers_sizes, expansion=1):
        stem = resnet_stem(3, 32, 32, 64)
        self.block_sizes = [64, 64, 128, 256, 512]

        for i in range(1, 5):
            self.block_sizes[i] *= expansion

        blocks = [self._make_layer(*o) for o in enumerate(layers_sizes)]

        super().__init__(*stem, *blocks,
                         nn.AdaptiveAvgPool2d(1), Flatten(),
                         nn.Linear(self.block_sizes[-1], out_classes))

    def _make_layer(self, idx, n_layers):
        stride = 1 if idx == 0 else 2
        ch_in, ch_out = self.block_sizes[idx:idx + 2]
        return nn.Sequential(*[
            ResBlock(ch_in if i == 0 else ch_out, ch_out, stride if i == 0 else 1)
            for i in range(n_layers)
        ])
