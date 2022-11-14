import torch
from torch import nn
import torch.nn.functional as F

class SGSELayer(nn.Module):
    def __init__(self, channel, reduction=16, with_sigmoid=True, groups=None, mode='noG_yesD'):
        self.groups = groups
        self.mode = mode

        super(SGSELayer, self).__init__()
        if self.mode == 'noG_yesD':
            self.num_SEpara = 3

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
        for d in range(3):
            _d = int(input_shape[d] / self.groups)
            if input_shape[d] % self.groups != 0:
                print(d, input_shape[d])
            xs = torch.split(x, split_size_or_sections=_d, dim=2 + d)

            # _xs_sequences = []
            _ys_sequences = []
            for g in range(self.groups):
                y = self.avg_pool(xs[g]).view(b, c)
                if self.mode == 'noG_yesD':
                    y = self.fc[d](y).view(b, c, 1)
                _ys_sequences.append(y)
            _y_sequences.append(_ys_sequences)

        return _y_sequences