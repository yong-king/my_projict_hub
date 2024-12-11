import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import pywt
import pywt.data
from ultralytics.nn.modules.block import Conv


class Attnbllock(nn.Module):
    def __init__(self, input_channels, output_channels, bias=False):
        super(Attnbllock, self).__init__()

        # 根据YOLO的输入调整输入维度
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.conv1 = Conv(input_channels, 2 * self.in_channels, 1)
        # 1x1 卷积代替全连接层
        self.conv_fc1 = Conv(input_channels, 192, 1)
        self.conv_fc2 = Conv(input_channels, 192, 1)

        # 深度可分离卷积
        self.depthwise_separable_conv1 = Conv(64, 128, 3, 1, 1, 64)
        self.depthwise_separable_conv2 = Conv(64, 128, 3, 1, 1,64)
        self.depthwise_separable_conv3 = Conv(64, 256, 3, 1, 1, 64)

        # AdaptiveAvgPool2d
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # # Swish激活函数
        self.swish = self.swish_activation
        self.tanh = self.tanh_activation

        # 定义 fc1 和 fc2 层
        self.fc1 = Conv(128, 256, 3, 1, 1)  # 假设拼接后为 256*2 的维度
        self.fc2 = Conv(257, self.out_channels, 3, 1, 1)

    def forward(self, x, input_size):
        # 1. 对x进行LayerNorm，分为x1和x2
        norm = nn.BatchNorm2d(x.size(1)).to(x.device)
        x = norm(x)
        x_t = x.clone()
        b, c, h, w = x.size()
        x = self.conv1(x)
        x = x.reshape(b, 2, -1, h, w).transpose(0, 1).contiguous()
        x1, x2 = x


        # 2. 对x1进行处理
        # 2.1 对x1进行全连接后分为x11, x12, x13
        x1_out = self.conv_fc1(x1)
        b1, c1, h1, w1 = x1_out.size()
        x1_out = x1.reshape(b1, 3, -1, h1, w1).transpose(0, 1).contiguous()
        x11, x12, x13 = x1_out


        # 2.2 对x12, x13进行AdaptiveAvgPool2d
        x12 = self.avg_pool(x12)
        x13 = self.avg_pool(x13)


        x11 = torch.softmax(x11 * x12, dim=1)

        # 使用expand_as确保x13的尺寸与x11相同
        x1 = x11 * x13

        # 3. 对x2进行处理
        # 3.1 对x2进行全连接后分为x21, x22, x23
        x2_out = self.conv_fc2(x2)
        b2, c2, h2, w2 = x2_out.size()
        x2_out = x2_out.reshape(b2, 3, -1, h2, w2).transpose(0, 1).contiguous()
        x21, x22, x23 = x2_out

        # 3.2 对x21, x22进行DepthwiseSeparableConv，然后matmul得到x21
        x21 = self.depthwise_separable_conv1(x21)
        x22 = self.depthwise_separable_conv2(x22)
        # 3.3 对x23进行DepthwiseSeparableConv得到x23
        x23 = self.depthwise_separable_conv3(x23)
        x21 = x21 * x22

        # 3.4 将x21进行全连接得，Swish，全连接，Tanh得到x21
        x21 = self.swish(x21)
        x21 = self.fc1(x21)
        x21 = self.tanh(x21)

        # 3.5 将x21与x23进行matmul得到x2
        x2 = x21 * x23

        # 4. 将x1,x2进行cat得到x
        x = torch.cat((x1, x2), dim=1)

        # 5. 对x进行全连接
        x = self.fc2(x)
        # 6. 将x与x_t进行add
        x = x + x_t
        return x

    # 定义 Swish 激活函数
    @staticmethod
    def swish_activation(x):
        return x * torch.sigmoid(x)

    # 定义 Tanh 激活函数
    @staticmethod
    def tanh_activation(x):
        return torch.tanh(x)


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


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

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

        self.attention = Attnbllock(self.in_channels, self.in_channels)  # Add SEBlock attention mechanism
        # self.attention = EfficientAttention()

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=None, stride=self.stride,
                                       groups=self.in_channels)
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
        # Apply channel attention
        batch_size, channels, height, width = x.size()
        x = self.attention(x, input_size=(height, width)) + x_tag  # Pass input_size here

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

# # 创建一个示例输入张量
# # Batch size: 1, Channels: 3, Height: 64, Width: 64
# input_tensor = torch.randn(1, 3, 64, 64)
#
# # 实例化 WTConv2d
# # 输入通道和输出通道设置为3（与输入一致），小波分解层级为1，使用db1小波
# wtconv2d = WTConv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, wt_levels=1, wt_type='db1')
#
# # 将输入张量传递给 WTConv2d 模块
# output_tensor = wtconv2d(input_tensor)
#
# # 打印输出的形状
# print("Input shape:", input_tensor.shape)
# print("Output shape:", output_tensor.shape)
