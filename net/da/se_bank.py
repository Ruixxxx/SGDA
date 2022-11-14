from torch import nn
from configs.sgda_config import config as cfg

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.ModuleList(
                    [nn.Sequential(nn.Linear(channel, channel // reduction),
                     nn.ReLU(inplace=True),
                     nn.Linear(channel // reduction, channel)) for n_classes in cfg['num_class']]
        )
        self.activate = nn.Sigmoid()

    def forward(self, x, cls_ind):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc[cls_ind](y).view(b, c, 1, 1, 1)
        y = self.activate(y)
        return x * y