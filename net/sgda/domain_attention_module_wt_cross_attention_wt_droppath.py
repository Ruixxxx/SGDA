# 104_x_787 and 105, 112, 111, 130, 152, 140 use it
from torch import nn
import torch
from net.sgda.se_module_vector import SGSELayer
from configs.sgda_config import config as cfg

class DomainAttention(nn.Module):
    def __init__(self, planes, reduction=16, nclass_list=None, fixed_block=False, groups=None, mode='noG_yesD',
                 attention_dir = False, drop_path=0.3, net_mode='train'):
        self.groups = groups
        self.mode = mode
        self.attention_dir = attention_dir
        self.net_mode = net_mode

        super(DomainAttention, self).__init__()
        if self.mode == 'noG_yesD':
            self.num_SEpara = 3
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
        reduct_planes = int(planes / 2)
        if 'NoduleNet' in cfg['net_name']:
            self.cross_attention = cross_attention(planes, reduct_planes)
        elif 'SANet' in cfg['net_name']:
            self.cross_attention2 = cross_attention2(planes, reduct_planes)
        self.drop_path = DropPath(drop_path, self.net_mode) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        _x_sequences = []
        b, c, *input_shape = x.size()

        if self.fixed_block:
            SELayers_Matrix = self.SGSE_Layers[0](x).view(b, c, 1, 1, 1)
            SELayers_Matrix = self.sigmoid(SELayers_Matrix)
        else:
            if not self.attention_dir:
                weight = self.fc_1(self.avg_pool(x).view(b, c))
                weight = self.softmax(weight).view(b, self.n_datasets, 1)

            all_y_sequences = [self.SGSE_Layers[i](x) for i in range(self.n_datasets)]
            for d in range(3):
                _d = int(input_shape[d] / self.groups)
                if input_shape[d] % self.groups != 0:
                    print(d, input_shape[d])
                xs = torch.split(x, split_size_or_sections=_d, dim=2+d)

                _xs_sequences = []
                for g in range(self.groups):
                    if self.attention_dir:
                        weight = self.fc_1[d](self.avg_pool(xs[g]).view(b, c))
                        weight = self.softmax(weight).view(b, self.n_datasets, 1)
                    for i in range(self.n_datasets):
                        if i == 0:
                            y = all_y_sequences[i][d][g]
                        else:
                            y = torch.cat((y, all_y_sequences[i][d][g]), 2)
                    y = torch.matmul(y, weight).view(b, c, 1, 1, 1)
                    y = self.sigmoid(y)
                    _xs_sequences.append(xs[g] * y)
                _x_sequences.append(torch.cat(_xs_sequences, dim=2+d))
            # x = (_x_sequences[0] + _x_sequences[1] + _x_sequences[2]) / 3
            if 'NoduleNet' in cfg['net_name']:
                x = (_x_sequences[0] + _x_sequences[1] + _x_sequences[2]) / 3 + \
                    self.drop_path(self.cross_attention(_x_sequences[0], _x_sequences[1], _x_sequences[2]))
            elif 'SANet' in cfg['net_name']:
                x = (_x_sequences[0] + _x_sequences[1] + _x_sequences[2]) / 3 + \
                    0.1 * self.drop_path(self.cross_attention2(_x_sequences[0], _x_sequences[1], _x_sequences[2]))
        return x


class cross_attention2(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
        """
    def __init__(self, inplanes, planes, use_scale=False):
        self.use_scale = use_scale
        self.groups = cfg['groups']
        self.pool_size = 4

        super(cross_attention2, self).__init__()
        # conv theta
        self.t = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv3d(planes, inplanes, kernel_size=1, stride=1,
                           groups=self.groups, bias=False)
        # self.bn = nn.BatchNorm3d(inplanes, momentum=0.1)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
        self.max_pool = nn.AdaptiveMaxPool3d(self.pool_size)
        self.softmax1 = nn.Softmax(dim=2)

    def non_local(self, t, p, g, b, c, zs, h, w, zs2):
        t = t.contiguous().view(b, c, zs * h * w).permute(0, 2, 1)
        p = p.contiguous().view(b, c, zs2 * self.pool_size * self.pool_size)
        g = g.contiguous().view(b, c, zs2 * self.pool_size * self.pool_size).permute(0, 2, 1)

        att = torch.bmm(t, p)

        if self.use_scale:
            # att = att * ((c * d * h * w) ** -0.5)
            # att = att.div((d * h * w) ** 0.5)
            att = att - torch.max(att)

        att = self.softmax1(att)
        x = torch.bmm(att, g)
        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, zs, h, w)
        return x

    def forward(self, x1, x2, x3):

        t = self.t(x1)
        p = self.p(x2)
        p = self.max_pool(p)
        g = self.g(x3)
        g = self.max_pool(g)

        b, c, zs, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(zs / self.groups)
            _c2 = int(self.pool_size / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=2)
            ps = torch.split(p, split_size_or_sections=_c2, dim=2)
            gs = torch.split(g, split_size_or_sections=_c2, dim=2)

            _t_sequences = []
            for i in range(self.groups):
                # _x = self.kernel(ts[i], ps[i], gs[i],
                #                  b, c, _c, h, w)
                _x = self.non_local(ts[i], ps[i], gs[i], b, c, _c, h, w, _c2)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=2)
        # else:
        #     x = self.non_local(t, p, g,
        #                     b, c, _c, h, w, _c2)

        del t, p, g, ts, ps, gs, _t_sequences, _x
        x = self.z(x)
        x = self.gn(x)

        return x



class cross_attention(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """
    def __init__(self, inplanes, planes, use_scale=False):
        self.use_scale = use_scale
        self.pool_size = 4 #4

        super(cross_attention, self).__init__()
        # conv theta
        self.t = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv3d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.max_pool = nn.AdaptiveMaxPool3d(self.pool_size)
        self.softmax1 = nn.Softmax(dim=2)
        self.bn = nn.BatchNorm3d(inplanes, momentum=0.1)

    def forward(self, x1, x2, x3):
        t = self.t(x1)
        p = self.p(x2)
        p = self.max_pool(p)
        g = self.g(x3)
        g = self.max_pool(g)

        b, c, d, h, w = t.size()

        t = t.contiguous().view(b, c, d * h * w).permute(0, 2, 1)
        p = p.contiguous().view(b, c, self.pool_size * self.pool_size * self.pool_size)
        g = g.contiguous().view(b, c, self.pool_size * self.pool_size * self.pool_size).permute(0, 2, 1)

        att = torch.bmm(t, p)

        if self.use_scale:
            # att = att * ((c * d * h * w) ** -0.5)
            # att = att.div((d * h * w) ** 0.5)
            att = att - torch.max(att)

        att = self.softmax1(att)
        x = torch.bmm(att, g)
        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, d, h, w)
        x = self.z(x)
        x = self.bn(x)
        # x = self.bn(x) + x1

        return x

def drop_path(x, drop_prob, net_mode):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or net_mode != 'train':
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, net_mode='train'):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.net_mode = net_mode

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.net_mode)
