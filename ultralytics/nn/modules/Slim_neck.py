from .block import Conv
import torch
import torch.nn as nn
import math


# class GSConv(nn.Module):
#     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
#         super().__init__()
#         c_ = c2 // 2
#         self.cv1 = Conv(c1, c_, k, s, None, g, 1, act)
#         self.cv2 = Conv(c_, c_, 5, 1, None, c_, 1, act)
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = torch.cat((x1, self.cv2(x1)), 1)
#         b, n, h, w = x2.data.size()
#         b_n = b * n // 2
#         y = x2.reshape(b_n, 2, h * w)
#         y = y.permute(1, 0, 2)
#         y = y.reshape(2, -1, n // 2, h, w)
#
#         return torch.cat((y[0], y[1]), 1)
#
class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        # self.cv1 = Conv(c1, c_, k, s, None, g, 1, act)
        # self.cv2 = Conv(c_, c_, 5, 1, None, c_, 1, act)
        self.cv1 = Conv(c1, c1, k, s, None, g, 1, act)
        self.cv2 = Conv(2*c1, c_, k, s, None, g, 1, act)
        self.cv3 = Conv(c_, c_, 5, 1, None, c_, 1, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x1 = self.cv2(torch.cat((x, x1), dim=1))
        x2 = torch.cat((x1, self.cv3(x1)), 1)
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

# class GSBottleneck(nn.Module):
#     # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)
#         # for lighting
#         self.conv_lighting = nn.Sequential(
#             GSConv(c1, c_, 1, 1),
#             GSConv(c_, c2, 3, 1, act=False))
#         self.shortcut = Conv(c1, c2, 1, 1, act=False)
#
#     def forward(self, x):
#         return self.conv_lighting(x) + self.shortcut(x)
class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.25):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c_, 3, 1, act=False))
        self.shortcut = Conv(c1, c_, 1, 1, act=False)
        self.conv1 = Conv(c_, c_, 3, 1, 1, c1)
        self.conv2 = Conv(c_, c_, (1, 11), 1, 5, c1)
        self.conv3 = Conv(c_, c_, (11, 1), 1, 5, c1)

    def forward(self, x):
        # return self.conv_lighting(x) + self.shortcut(x)
        b, c, h, w = x.shape()
        x1, x2, x3, x4 = torch.split(x, (c/4, c/4, c/4, c/4), dim=1)
        return torch.cat(self.conv1(x1), self.conv2(x2), self.conv3(x3), self.conv_lighting(x)+self.shortcut(x))


class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)  #

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))





