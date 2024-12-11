
import torch.nn as nn
from functools import partial
import pywt
import pywt.data
import torch

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
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db2'):
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


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        if c_ == c2:
            self.cv2 = WTConv2d(c_, c2, 5, 1)
        else:
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
class SwishFunction(Function):
    """
    Memory-efficient implementation of Swish activation using custom autograd.
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: Compute Swish activation x * sigmoid(x).
        """
        ctx.save_for_backward(x)  # Save input for backward pass
        sigmoid_x = torch.sigmoid(x)
        return x * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradient of Swish.
        """
        x, = ctx.saved_tensors  # Retrieve saved input
        sigmoid_x = torch.sigmoid(x)
        grad_x = grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
        return grad_x

class MemoryEfficientSwish(nn.Module):
    """
    Wrapper module for memory-efficient Swish activation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SwishFunction.apply(x)

class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
                            nn.Conv2d(dim, dim, 1, 1, 0),
                            MemoryEfficientSwish(),
                            nn.Conv2d(dim, dim, 1, 1, 0)
                         )
    def forward(self, x):
        return self.act_block(x)


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, window_size=7,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.window_size =  window_size

        # Define WTConv2d for high-frequency attention
        self.wt_conv = WTConv2d(
            in_channels=3 * dim,
            out_channels=3 * dim
        )
        self.high_fre_qkv = nn.Conv2d(dim, 3 * dim, 1, bias=qkv_bias)
        self.high_fre_attn_block = AttnMap(dim)

        # Define layers for low-frequency attention
        self.global_q = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.global_kv = nn.Conv2d(dim, 2 * dim, 1, bias=qkv_bias)
        self.avgpool = nn.AvgPool2d(window_size) if window_size != 1 else nn.Identity()

        # Projection layers
        self.proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attention(self, x):
        """
        High-frequency attention: uses WTConv2d and attention block.
        """
        b, c, h, w = x.shape
        qkv = self.high_fre_qkv(x)  # (b, 3 * c, h, w)
        qkv = self.wt_conv(qkv).reshape(b, 3, c, h, w)  # Apply WTConv2d
        q, k, v = qkv.unbind(dim=1)  # Split into Q, K, V
        attn = self.high_fre_attn_block(q.mul(k)) * self.scalor
        attn = self.attn_drop(torch.tanh(attn))
        return attn.mul(v)  # (b, c, h, w)

    def low_fre_attention(self, x):
        """
        Low-frequency attention: uses average pooling and global QKV computation.
        """
        b, c, h, w = x.shape
        q = self.global_q(x).reshape(b, self.num_heads, self.dim_head, -1).permute(0, 1, 3, 2)  # (b, num_heads, h*w, dim_head)
        kv = self.avgpool(x)
        kv = self.global_kv(kv).reshape(b, 2, self.num_heads, self.dim_head, -1).permute(1, 0, 2, 4,  3)
        k, v = kv  # Split into K, V
        attn = (q @ k.transpose(-1, -2)) * self.scalor  # (b, num_heads, h*w, h*w)
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = (attn @ v).permute(0, 1, 3, 2).reshape(b, c, h, w)  # (b, c, h, w)
        return res


    def forward(self, x):
        """
        Compute both high-frequency and low-frequency attention and combine them.
        """
        high_fre = self.high_fre_attention(x)
        low_fre = self.low_fre_attention(x)
        combined = high_fre + low_fre  # Combine results
        return self.proj_drop(self.proj(combined))  # Project output

if __name__ == '__main__':
    attention = EfficientAttention(dim=64, num_heads=4, kernel_size=3, window_size=7)
    x = torch.randn(1, 64, 32, 32)  # Example input
    output = attention(x)
    print(output.shape)  # Expected output: (8, 64, 32, 32)



