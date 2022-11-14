# 104_x_787 and 105, 112, 111, 130, 152, 140 use it
from torch import nn
import torch
from net.sgda.se_module_vector2 import SGSELayer
from configs.sgda_config import config as cfg

class DomainAttention(nn.Module):
    def __init__(self, planes, reduction=16, nclass_list=None, fixed_block=False, groups=None, mode='noG_1D',
                 attention_dir = False):
        self.groups = groups
        self.mode = mode
        self.attention_dir = attention_dir

        super(DomainAttention, self).__init__()
        if self.mode == 'noG_1D':
            self.num_SEpara = 1
        self.planes = planes
        num_adapters = cfg['num_adapters']
        if num_adapters == 0:
            self.n_datasets = len(nclass_list)
        else:
            self.n_datasets = num_adapters
        self.fixed_block = fixed_block
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # if not self.fixed_block and cfg.less_blocks:
        #     if cfg.block_id != 4:
        #         if cfg.layer_index % 2 == 0:
        #             self.fixed_block = True
        #     else:
        #         if cfg.layer_index % 2 != 0:
        #             self.fixed_block = True
        if self.fixed_block or num_adapters == 1:
            self.SGSE_Layers = nn.ModuleList([SGSELayer(planes, reduction,
                 with_sigmoid=False, groups=self.groups, mode=self.mode) for num_class in range(1)])
        elif num_adapters == 0:
            self.SGSE_Layers = nn.ModuleList([SGSELayer(planes, reduction,
                 with_sigmoid=False, groups=self.groups, mode=self.mode) for num_class in nclass_list])
        else:
            self.SGSE_Layers = nn.ModuleList([SGSELayer(planes, reduction,
                 with_sigmoid=False, groups=self.groups, mode=self.mode) for num_class in range(num_adapters)])
        if self.attention_dir:
            self.fc_1 = nn.ModuleList([nn.Linear(planes, self.n_datasets) for i in range(self.num_SEpara)])
        else:
            self.fc_1 = nn.Linear(planes, self.n_datasets)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _x_sequences = []
        b, c, *input_shape = x.size()

        if self.fixed_block:
            SELayers_Matrix = self.SGSE_Layers[0](x).view(b, c, 1, 1, 1)
            SELayers_Matrix = self.sigmoid(SELayers_Matrix)
        else:
            # if not self.attention_dir:
            #     weight = self.fc_1(self.avg_pool(x).view(b, c))
            #     weight = self.softmax(weight).view(b, self.n_datasets, 1)

            all_y_sequences = [self.SGSE_Layers[i](x) for i in range(self.n_datasets)]
            for d in range(1):
                d = d + cfg['d']
                _d = int(input_shape[d] / self.groups)
                if input_shape[d] % self.groups != 0:
                    print(d, input_shape[d])
                xs = torch.split(x, split_size_or_sections=_d, dim=2+d)

                _xs_sequences = []
                for g in range(self.groups):
                    if self.attention_dir:
                        weight = self.fc_1[0](self.avg_pool(xs[g]).view(b, c))
                        weight = self.softmax(weight).view(b, self.n_datasets, 1)
                    else:
                        weight = self.fc_1(self.avg_pool(xs[g]).view(b, c))
                        weight = self.softmax(weight).view(b, self.n_datasets, 1)
                    for i in range(self.n_datasets):
                        if i == 0:
                            y = all_y_sequences[i][0][g]
                        else:
                            y = torch.cat((y, all_y_sequences[i][0][g]), 2)
                    y = torch.matmul(y, weight).view(b, c, 1, 1, 1)
                    y = self.sigmoid(y)
                    _xs_sequences.append(xs[g] * y)
                _x_sequences.append(torch.cat(_xs_sequences, dim=2+d))
            x = _x_sequences[0]

        return x