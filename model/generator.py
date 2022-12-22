import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

class ResidiumBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convolutions = nn.ModuleList()
        for i in range(len(dilation)):
            self.convolutions.append(weight_norm(Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[i],
                padding=(kernel_size * dilation[i] - dilation[i]) // 2
            )))
        self.convolutions.apply(init_weights)

    def forward(self, x):
        for conv in self.convolutions:
            x = conv(F.leaky_relu(x, 0.1)) + x
        return x

class Generator(torch.nn.Module):
    def __init__(self, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_kernel_sizes, upsample_initial_channel):
        super().__init__()
        self.pre_convolution = weight_norm(Conv1d(80, upsample_initial_channel, 7, 1, 3))

        self.up_blocks = nn.ModuleList()
        cur_channel = upsample_initial_channel
        for rate, kernel in zip(upsample_rates, upsample_kernel_sizes):
            self.up_blocks.append(weight_norm(ConvTranspose1d(
                cur_channel, cur_channel // 2, kernel, rate, (kernel - rate) // 2
            )))
            cur_channel //= 2
        self.up_blocks.apply(init_weights)

        self.num_kernels = len(resblock_kernel_sizes)
        self.resblocks = nn.ModuleList()
        cur_channel = upsample_initial_channel
        for i in range(len(self.up_blocks)):
            cur_channel //= 2
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResidiumBlock(cur_channel, kernel, dilation))

        self.post_convolution = weight_norm(Conv1d(cur_channel, 1, 7, 1, 3))
        self.post_convolution.apply(init_weights)

    def forward(self, x):
        x = self.pre_convolution(x)
        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](F.leaky_relu(x, 0.1))
            res_x = self.resblocks[i * self.num_kernels](x)
            for j in range(self.num_kernels):
                res_x += self.resblocks[i * self.num_kernels + j](x)
            x = res_x / self.num_kernels
        x = F.leaky_relu(x)
        x = self.post_convolution(x)
        x = torch.tanh(x)

        return x
