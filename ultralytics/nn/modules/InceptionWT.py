from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
import pywt


class wavelet():
    def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
        w = pywt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
        dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                                   dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                                   dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                                   dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

        dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

        rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
        rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
        rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

        rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

        return dec_filters, rec_filters

    def wavelet_transform(x, filters):
        b, c, h, w = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
        x = x.reshape(b, c, 4, h // 2, w // 2)
        return x

    def inverse_wavelet_transform(x, filters):
        b, c, _, h_half, w_half = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = x.reshape(b, c * 4, h_half, w_half)
        x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
        return x
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet.wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(wavelet.inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


from functools import partial
import torch.nn as nn
from timm.models.layers import to_2tuple
from .block import Conv

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = Conv(gc, gc, square_kernel_size, p=square_kernel_size // 2, g=gc)
        self.dwconv_w = Conv(gc, gc, k=(1, band_kernel_size), p=(0, band_kernel_size // 2),
                                  g=gc)
        self.dwconv_h = Conv(gc, gc, k=(band_kernel_size, 1), p=(band_kernel_size // 2, 0),
                                  g=gc)
        self.conv_wt = WTConv2d(in_channels - 3 * gc, in_channels - 3 * gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (self.conv_wt(x_id), self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features
        hidden_features = in_features
        # bias = to_2tuple(bias)

        self.fc1 = Conv(in_features, hidden_features, k=1)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = Conv(hidden_features, out_features, k=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000, mlp_ratio=3, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.mean((2, 3))  # global average pooling
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class InceptionNeXt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inception = InceptionDWConv2d(in_channels)
        self.mlp = ConvMlp(in_channels, out_channels)

    def forward(self, x):
        x_t = x.clone()
        x = self.mlp(self.inception(x))
        return x + x_t

# class InceptionNeXt(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
#         super().__init__()
#     # def Token_mixer(self, in_channels, hidden_channels, out_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
#         self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
#
#         gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
#         self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
#         self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
#                                   groups=gc)
#         self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
#                                   groups=gc)
#         self.wtconv = WTConv2d(in_channels - 3 * gc, in_channels - 3 * gc)
#         self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
#         self.norm_layer1 = nn.BatchNorm2d(in_channels)
#     # def ConvMLP(self, in_channels, hidden_channels, out_channels):    # ConvMLP
#         self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1,stride=1)
#         self.norm = nn.BatchNorm2d(hidden_channels)
#         self.act = nn.GELU()
#         self.drop = nn.Dropout(p=0.0, inplace=False)
#         self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1)
#     def forward(self, x):
#         x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
#         x_hw = self.dwconv_hw(x_hw)
#         x_w = self.dwconv_w(x_w)
#         x_h = self.dwconv_h(x_h)
#         x_id = self.wtconv(x_id)
#         split_x = torch.cat((x_id, x_hw, x_w, x_h), dim=1 ) + x
#         mixer_out = self.norm_layer1(split_x)
#
#         convmlp_out = self.fc1(mixer_out)
#         convmlp_out = self.norm(self.act(self.drop(convmlp_out)))
#         convmlp_out = self.fc2(convmlp_out)
#         x = self.adjust_channels(x)
#         output = convmlp_out + x
#         return output



import torch
import torch.nn as nn


# 假设你已经导入了上面定义的所有类，包括 InceptionDWConv2d, ConvMlp, MlpHead, InceptionNeXt 等
#
# 测试用例
# def test_inception_next():
#     # 设置随机种子确保结果可复现
#     torch.manual_seed(42)
#
#     # 假设输入的通道数为 64，且输入图像大小为 32x32
#     in_channels = 64
#     batch_size = 8
#     height, width = 32, 32
#
#     # 创建一个模拟输入张量，形状为 (batch_size, in_channels, height, width)
#     x = torch.randn(batch_size, in_channels, height, width)
#
#     # 创建模型实例
#     model = InceptionNeXt(in_channels=in_channels)
#
#     # 打印模型结构（可选）
#     print(model)
#
#     # 前向传播测试
#     output = model(x)
#
#     # 输出张量形状检查
#     print("Output shape:", output.shape)
#     assert output.shape == (batch_size, in_channels, height, width), "Output shape mismatch"
#
#     print("Test passed successfully!")


# 运行测试用例
# test_inception_next()
