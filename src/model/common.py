import math

import torch
import torch.nn as nn
import torch.nn.functional as F



def round_filters(filters, multiplier=1.0, divisor=8, min_depth=None):
    multiplier = multiplier
    divisor = divisor
    min_depth = min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters




def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True, diliation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride,
        padding=(kernel_size//2), bias=bias, dilation=diliation)

def default_dconv(in_channels, out_channels, kernel_size, dil=2, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, dilation= dil,
        padding=(kernel_size//2)+1, bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range,rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class rhcnnba(nn.Module):
    def __init__(
        self, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(rhcnnba, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.branch =nn.Sequential(
            nn.Conv2d(n_feat, n_feat//4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(True),
            wn(nn.Conv2d(n_feat//4, n_feat//4, kernel_size=3, stride=1,padding=(kernel_size//2),bias=bias)),
        )
    def forward(self, x):
        x0 = self.branch(x)
        x1 = self.branch(x)
        x2 = self.branch(x)
        x3 = self.branch(x)
        out = torch.cat((x0, x1,x2,x3), 1)

        out += x

        return out

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=True):

        m = [conv(in_channels, out_channels, kernel_size, stride=stride, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act:
            m.append(nn.ReLU(True))

        super(BasicBlock, self).__init__(*m)

class DepthWiseWithSkipBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, reduction=1):
        super(DepthWiseWithSkipBlock, self).__init__()
        self.expansion = 1 / float(reduction)
        self.in_planes = in_planes
        self.mid_planes = mid_planes = int(self.expansion * out_planes)
        self.out_planes = out_planes

        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.depth = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, padding=1,
                               stride=1, bias=False, groups=mid_planes)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(
            mid_planes, out_planes, kernel_size=1, bias=False, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (
        _, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h * int_w * self.mid_planes * self.in_planes + out_h * out_w * self.mid_planes * self.out_planes
        flops += out_h * out_w * self.mid_planes * 9  # depth-wise convolution
        if len(self.shortcut) > 0:
            flops += self.in_planes * self.out_planes * out_h * out_w
        return flops

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = out.size()
        out = self.bn2(self.depth(out))
        out = self.bn3(self.conv3(out))
        self.out_nchw = out.size()
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]
class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)
            #self.square_bn = FilterResponseNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,padding=ver_conv_padding)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,padding=hor_conv_padding)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)

            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)



    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)

            square_outputs = self.square_bn(square_outputs)
            #print(square_outputs.size())
            # return square_outputs
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)

            vertical_outputs = nn.AdaptiveMaxPool2d((square_outputs.cpu().detach().numpy().shape[2], square_outputs.cpu().detach().numpy().shape[3]))(
                vertical_outputs)
            #print(vertical_outputs.size())
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)

            horizontal_outputs = nn.AdaptiveMaxPool2d(
                (square_outputs.cpu().detach().numpy().shape[2], square_outputs.cpu().detach().numpy().shape[3]))(
                horizontal_outputs)
            #print(horizontal_outputs.size())
            return square_outputs + vertical_outputs + horizontal_outputs
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class DenseBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size, padding, bias=True):
        super(DenseBlock, self).__init__()

        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=padding,bias=bias)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(n_feats*2, n_feats, kernel_size, padding=padding,bias=bias)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(n_feats*3, n_feats, kernel_size, padding=padding,bias=bias)
        self.prelu3 =nn.PReLU()

        self.conv4 = nn.Conv2d(n_feats*4, n_feats, kernel_size, padding=padding,bias=bias)
        self.prelu4 =nn.PReLU()

        # Local Feature Fusion
        self.conv5 = nn.Conv2d(n_feats*5, n_feats, kernel_size=1, padding=0,bias=bias)
        self.prelu5 =nn.PReLU()


    def forward(self, input):
        x = input
        x1 = self.prelu1(self.conv1(x))
        x_cat1 = torch.cat((x, x1), dim=1)
        x2 = self.prelu2(self.conv2(x_cat1))
        x_cat2 = torch.cat((x, x1, x2), dim=1)
        x3 = self.prelu3(self.conv3(x_cat2))
        x_cat3 = torch.cat((x, x1, x2, x3), dim=1)
        x4 = self.prelu4(self.conv4(x_cat3))
        x_cat4 = torch.cat((x, x1, x2, x3, x4), dim=1)
        x5 = self.prelu5(self.conv5(x_cat4))
        output = torch.add(input, x5)

        return output



class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # self.conv0 = nn.Conv2d(96, 96, 9, 1, 4)
        # self.prelu0 = nn.PReLU()
        #self.conv1 = nn.Conv2d(64, 128, 7, 1, 3)
        self.conv1 = ACBlock(64, 128, 7, 1, 3)
        self.prelu1 = nn.PReLU()
        #self.conv2 = nn.Conv2d(64, 128, 5, 2, 2)
        self.conv2 = ACBlock(64, 128, 5, 2, 2)
        self.prelu2 = nn.PReLU()

        #self.conv3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv3 = ACBlock(128, 128, 3, 2, 1)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.up1 = nn.PixelShuffle(2)
        self.prelu4 =nn.PReLU()

        self.conv5 = nn.Conv2d(192, 96, 3, 1, 1)
        self.prelu5 = nn.PReLU()

        self.conv6 = nn.Conv2d(128, 256, 3, 1, 1)
        self.up2 = nn.PixelShuffle(2)
        self.prelu6 = nn.PReLU()

        self.conv7 = nn.Conv2d(192, 96, 3, 1, 1)
        self.prelu7 = nn.PReLU()


        self.conv10 = nn.Conv2d(128, 256, 3, 1, 1)
        self.up4 = nn.PixelShuffle(2)
        self.prelu10 = nn.PReLU()

        self.conv11 = nn.Conv2d(320, 128, 3, 1, 1)
        self.prelu11 = nn.PReLU()

        self.block8 = DepthWiseWithSkipBlock(128, 128)
        self.block9 = DepthWiseWithSkipBlock(96, 128)
        self.block10 = DepthWiseWithSkipBlock(96, 128)


        self.conv12 = nn.Conv2d(128, 64, 3, 1, 1)

    def forward(self, input):

        x = input
        #x0 = self.conv1(x)
        #x1 = self.conv2(x)
        #x2 = self.conv3(x1)
        x0 = self.prelu1(self.conv1(x))
        x1 = self.prelu2(self.conv2(x))
        x2 = self.prelu3(self.conv3(x1))

        xt1 = self.block8(x2)
        x_cat1 = self.prelu4(self.up1(self.conv4(xt1)))

        x = torch.cat((x_cat1, x1), dim=1)
        x = self.prelu5(self.conv5(x))

        xt2 = self.block9(x)
        x_cat2 = self.prelu6(self.up2(self.conv6(xt2)))
        x = torch.cat((x_cat2,x0), dim=1)
        x = self.prelu7(self.conv7(x))


        xt3 = self.block10(x)
        x_cat4 = self.prelu10(self.conv10(xt3))
        x = torch.cat((x_cat4, input), dim=1)
        x = self.prelu7(self.conv11(x))

        output = self.conv12(x)
        return output



class MyNet2(nn.Module):
    def __init__(self):
        super(MyNet2, self).__init__()

        # self.conv0 = nn.Conv2d(96, 96, 9, 1, 4)
        # self.prelu0 = nn.PReLU()
        self.conv1 = nn.Conv2d(96, 96, 7, 1, 3)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(96, 96, 5, 2, 2)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(96, 96, 3, 2, 1)
        self.prelu3 = nn.PReLU()

        self.conv4 = rhcnnba(96, 3)
        self.up1 = nn.PixelShuffle(2)
        self.prelu4 =nn.PReLU()

        self.conv5 = nn.Conv2d(120, 96, 3, 1, 1)
        self.prelu5 = nn.PReLU()

        self.conv6 = rhcnnba(96, 3)
        self.up2 = nn.PixelShuffle(2)
        self.prelu6 = nn.PReLU()

        self.conv7 = nn.Conv2d(120, 96, 3, 1, 1)
        self.prelu7 = nn.PReLU()

        # self.conv8 = nn.Conv2d(96, 96, 3, 1, 1)
        # self.up3 = nn.PixelShuffle(2)
        # self.prelu8 = nn.PReLU()
        #
        # self.conv9 = nn.Conv2d(120, 96, 3, 1, 1)
        # self.prelu9 = nn.PReLU()

        self.conv10 = rhcnnba(96, 3)
        self.up4 = nn.PixelShuffle(2)
        self.prelu10 = nn.PReLU()

        self.conv11 = nn.Conv2d(120, 96, 3, 1, 1)
        self.prelu11 = nn.PReLU()

        #self.block8 = DepthWiseWithSkipBlock(96, 96)
        self.block8 = rhcnnba(96, 3)
        #self.block9 = DepthWiseWithSkipBlock(96, 96)
        self.block9 = rhcnnba(96, 3)
        #self.block10 = DepthWiseWithSkipBlock(96, 96)
        self.block10 = rhcnnba(96, 3)

        self.block11 = DepthWiseWithSkipBlock(96, 96)

        self.block12 = DepthWiseWithSkipBlock(96, 96)

        self.block13 = DepthWiseWithSkipBlock(96, 96)

        self.blockFinal = nn.Conv2d(96, 96, 3, 1, 1)

    def forward(self, input):

        x = input
        #9
        #x00 = self.prelu0(self.conv0(x))
        #7
        x0 = self.prelu1(self.conv1(x))
        #5
        x1 = self.prelu2(self.conv2(x))
        #3
        x2 = self.prelu3(self.conv3(x1))

        #x = self.block8(x2)
        #x = self.block11(x)
        xt1 = self.block10(x2)
        x_cat1 = self.prelu4(self.up1(self.conv4(xt1)))
        x_cat1 = nn.AdaptiveMaxPool2d((x1.cpu().detach().numpy().shape[2],x1.cpu().detach().numpy().shape[3]))(x_cat1)
        x = torch.cat((x_cat1, x1), dim=1)
        x = self.prelu5(self.conv5(x))

        #x = self.block8(x)
        #x = self.block11(x)
        xt2 = self.block10(x)
        x_cat2 = self.prelu6(self.up2(self.conv6(xt2)))
        x_cat2 = nn.AdaptiveMaxPool2d((x0.cpu().detach().numpy().shape[2], x0.cpu().detach().numpy().shape[3]))(x_cat2)
        x = torch.cat((x_cat2,x0), dim=1)
        x = self.prelu7(self.conv7(x))

        # x_cat3 = self.prelu8(self.up3(self.conv8(x)))
        # x_cat3 = nn.AdaptiveMaxPool2d((x00.cpu().detach().numpy().shape[2], x00.cpu().detach().numpy().shape[3]))(x_cat3)
        # x = torch.cat((x_cat3, x00), dim=1)
        # x = self.prelu7(self.conv9(x))
        #x = self.block13(x)
        #x = self.block8(x)
        #x = self.block11(x)
        xt3 = self.block10(x)
        x_cat4 = self.prelu10(self.up4(self.conv10(xt3)))
        x_cat4 = nn.AdaptiveMaxPool2d((input.cpu().detach().numpy().shape[2], input.cpu().detach().numpy().shape[3]))(x_cat4)
        x = torch.cat((x_cat4, input), dim=1)
        x = self.prelu7(self.conv11(x))
        #x = self.block11(x)

        output = self.blockFinal(x)
        return output


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                #m.append(conv(n_feats, 4 * n_feats))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            #m.append(conv(n_feats, 9 * n_feats))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# --------------------------------------------
# inverse of pixel_shuffle
# --------------------------------------------
def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    written by: Zhaoyi Yan, https://github.com/Zhaoyi-Yan
    and Kai Zhang, https://github.com/cszn/FFDNet
    01/01/2019
    """
    batch_size, channels, in_height, in_width = input.size()


    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    written by: Zhaoyi Yan, https://github.com/Zhaoyi-Yan
    and Kai Zhang, https://github.com/cszn/FFDNet
    01/01/2019
    """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

# spatial attention



