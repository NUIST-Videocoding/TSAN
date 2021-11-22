from model import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return TSAN(args)


#3*3 conv repalce 5*5 conv
class conv5(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=True):

        m = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=(kernel_size // 2), bias=bias),
            nn.PReLU(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=(kernel_size // 2), bias=bias),

        ]

        super(conv5, self).__init__(*m)

# Spatial attention
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.s1 = common.BasicBlock(conv, 2, 1, kernel_size=3, bn=True, act=False)
        self.s2 = common.BasicBlock(conv, 2, 1, kernel_size=5, bn=True, act=False)
        self.s3 = common.BasicBlock(conv, 2, 1, kernel_size=7, bn=True, act=False)
        self.s4 = common.BasicBlock(conv, 3, 1, kernel_size=1, bn=False, act=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        s1 = self.s1(x_compress)
        s2 = self.s2(x_compress)
        s3 = self.s3(x_compress)
        x_out = torch.cat([s1, s2, s3], 1)
        x_out = self.s4(x_out)
        scale = self.sig(x_out) # broadcasting
        return x * scale

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class MSRB(nn.Module):
    def __init__(self, conv=common.default_conv, n_feats=64):
        super(MSRB, self).__init__()

        kernel_size_1 = 3

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv5(n_feats, n_feats, kernel_size_1)
        self.conv_5_2 = conv5(n_feats * 2, n_feats * 2, kernel_size_1)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.prelu = nn.PReLU()
        self.att = SpatialGate(conv)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.prelu(self.conv_3_1(input_1))
        output_5_1 = self.prelu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.prelu(self.conv_3_2(input_2))
        output_5_2 = self.prelu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output = self.att(output)
        output += x
        return output

class TSAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(TSAN, self).__init__()
        n_feats = 64
        kernel_size = 3
        n_resblocks = 12
        act = nn.PReLU()

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_branch_down = [
            ResBlock(
                conv, n_feats, kernel_size, act=act)
            for _ in range(n_resblocks)
        ]
        modules_branch = [
            MSRB(
                conv, n_feats)
            for _ in range(6)
        ]

        # define tail module
        modules_tail = [
            conv(n_feats, args.n_colors, kernel_size)
            ]


        self.head = nn.Sequential(*modules_head)

        # pixelunshuffle part
        self.downsample = common.PixelUnShuffle(2)
        self.conv1 = conv(12, n_feats, 3)

        self.branch = nn.Sequential(*modules_branch)
        self.branch_down = nn.Sequential(*modules_branch_down)
        self.upsample = common.Upsampler(conv, 2, n_feats, act=False)
        self.confusion = nn.Conv2d(n_feats * 3, n_feats, 1, padding=0, stride=1)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x_down = self.downsample(x)
        x_down = self.conv1(x_down)
        x = self.head(x)
        res = x
        branch = self.branch(x)
        branch_down = self.branch_down(x_down)
        branch_down = self.upsample(branch_down)
        concat = torch.cat([branch_down, branch, res], 1)
        concat = self.confusion(concat)
        x = self.tail(concat)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
