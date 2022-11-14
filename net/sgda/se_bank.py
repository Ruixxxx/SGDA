import torch
from torch import nn
from configs.sgda_config import config as cfg

class SGSELayer(nn.Module):
    def __init__(self, channel, reduction=16, groups=None, mode='noG_yesD'):
        self.groups = groups
        self.mode = mode

        super(SGSELayer, self).__init__()
        if self.mode == 'noG_yesD':
            self.num_SEpara = 3

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.ModuleList([nn.ModuleList([
                     nn.Sequential(nn.Linear(channel, channel // reduction),
                     nn.ReLU(inplace=True),
                     nn.Linear(channel // reduction, channel)) for i in range(self.num_SEpara)])
                     for n_classes in cfg['num_class']]
        )
        self.activate = nn.Sigmoid()

    def forward(self, x, cls_ind):
        _x_sequences = []
        b, c, *input_shape = x.size()
        for d in range(3):
            _d = int(input_shape[d] / self.groups)
            if input_shape[d] % self.groups != 0:
                print(d, input_shape[d])
            xs = torch.split(x, split_size_or_sections=_d, dim=2 + d)

            _xs_sequences = []
            for g in range(self.groups):
                y = self.avg_pool(xs[g]).view(b, c)
                if self.mode == 'noG_yesD':
                    y = self.fc[cls_ind][d](y).view(b, c, 1, 1, 1)
                    y = self.activate(y)
                _xs_sequences.append(xs[g] * y)
            _x_sequences.append(torch.cat(_xs_sequences, dim=2 + d))
        x = (_x_sequences[0] + _x_sequences[1] + _x_sequences[2]) / 3

        return x