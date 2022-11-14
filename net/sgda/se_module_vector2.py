import torch
from torch import nn
from configs.sgda_config import config as cfg

class SGSELayer(nn.Module):
    def __init__(self, channel, reduction=16, with_sigmoid=True, groups=None, mode='noG_1D'):
        self.groups = groups
        self.mode = mode

        super(SGSELayer, self).__init__()
        if self.mode == 'noG_1D':
            self.num_SEpara = 1

        self.with_sigmoid = with_sigmoid
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        if with_sigmoid:
            self.fc = nn.ModuleList([nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()) for i in range(self.num_SEpara)])
        else:
            self.fc = nn.ModuleList([nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel)
                ) for i in range(self.num_SEpara)])

    def forward(self, x):
        _y_sequences = []
        b, c, *input_shape = x.size()
        for d in range(1):
            d = d + cfg['d']
            _d = int(input_shape[d] / self.groups)
            if input_shape[d] % self.groups != 0:
                print(d, input_shape[d])
            xs = torch.split(x, split_size_or_sections=_d, dim=2 + d)

            # _xs_sequences = []
            _ys_sequences = []
            for g in range(self.groups):
                y = self.avg_pool(xs[g]).view(b, c)
                if self.mode == 'noG_1D':
                    y = self.fc[0](y).view(b, c, 1)
                _ys_sequences.append(y)
            _y_sequences.append(_ys_sequences)

        return _y_sequences