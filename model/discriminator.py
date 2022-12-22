import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm


class PeriodDiscriminator(torch.nn.Module):
    def __init__(self, period, n_channels=[1, 32, 128, 512, 1024]):
        super().__init__()
        self.period = period
        self.convolutions = nn.ModuleList()
        for i in range(len(n_channels) - 1):
            self.convolutions.append(weight_norm(Conv2d(n_channels[i], n_channels[i+1], (5, 1), (3, 1), padding=(2, 0))))
        self.convolutions.append(weight_norm(Conv2d(n_channels[-1], n_channels[-1], (5, 1), 1, padding=(2, 0))))
        self.post_convolution = weight_norm(Conv2d(n_channels[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        features = []

        if x.shape[-1] % self.period != 0:
            diff = self.period - (x.shape[-1] % self.period)
            x = F.pad(x, (0, diff), "reflect")

        x = x.view(x.shape[0], x.shape[1], x.shape[2] // self.period, self.period)

        for conv in self.convolutions:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.post_convolution(x)
        features.append(x)
        
        x = torch.flatten(x, 1, -1)
        return x, features


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.period_discriminators = nn.ModuleList([PeriodDiscriminator(period) for period in periods])

    def forward(self, x, gen_x):
        outputs = []
        gen_outputs = []
        features = []
        gen_features = []
        for period_discriminator in self.period_discriminators:
            output, feature = period_discriminator(x)
            gen_output, gen_feature = period_discriminator(x)
            outputs.append(output)
            gen_outputs.append(gen_output)
            features.append(feature)
            gen_features.append(gen_feature)

        return outputs, gen_outputs, features, gen_features

class ScaleDiscriminator(torch.nn.Module):
    def __init__(self, norm_func=weight_norm):
        super().__init__()
        self.convolutions = nn.ModuleList([
            norm_func(Conv1d(1, 128, 15, 1, padding=7)),
            norm_func(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_func(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_func(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_func(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_func(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_func(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post_convolution = norm_func(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        features = []
        for conv in self.convolutions:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.post_convolution(x)
        features.append(x)
        
        x = torch.flatten(x, 1, -1)
        return x, features


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_discriminators = nn.ModuleList([
            ScaleDiscriminator(norm_func=spectral_norm),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.avg_pool = AvgPool1d(4, 2, 2)

    def forward(self, x, gen_x):
        outputs = []
        gen_outputs = []
        features = []
        gen_features = []
        for scale_discriminator in self.scale_discriminators:
            output, feature = scale_discriminator(x)
            gen_output, gen_feature = scale_discriminator(gen_x)
            outputs.append(output)
            features.append(feature)
            gen_outputs.append(gen_output)
            gen_features.append(gen_feature)
            x = self.avg_pool(x)
            gen_x = self.avg_pool(gen_x)

        return outputs, gen_outputs, features, gen_features
