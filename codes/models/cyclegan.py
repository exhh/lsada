import torch
import torch.nn as nn
from .models import register_model

class ResidualBlock(nn.Module):
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x

class ContractingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2)
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.conv(x)
        return x

class MatchFeatureMap(nn.Module):
    def __init__(self, mode='nearest', input_channels=3):
        super(MatchFeatureMap, self).__init__()
        self.interp = nn.functional.interpolate
        self.mode = mode

    def forward(self, x, desired_size):
        x = self.interp(x, size=desired_size, mode=self.mode)
        return x

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=64, use_tanh=True):
        super(Generator, self).__init__()
        self.use_tanh = use_tanh
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.matchfeature = MatchFeatureMap()
        if use_tanh:
            self.output = torch.nn.Tanh()
        else:
            self.output = torch.nn.Sigmoid()

    def forward(self, x):
        x_size = x.size()[-2:]
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand2(x11)
        x13 = self.expand3(x12)
        xn = self.downfeature(x13)
        xn = self.matchfeature(xn, x_size)
        return self.output(xn)

class Discriminator(nn.Module):
    '''
    This is from the official CycleGAN implementation:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/cf4191a3a4cc77fdffa5c0a8246c346049958e78/models/networks.py#L538
    '''
    def __init__(self, input_channels, hidden_channels=64, n_layers=3):
        super(Discriminator, self).__init__()
        kernel_size = 4
        padding_size = 1
        sequence = [nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=2, padding=padding_size), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(hidden_channels * nf_mult_prev, hidden_channels * nf_mult, kernel_size=kernel_size, stride=2, padding=padding_size),
                nn.InstanceNorm2d(hidden_channels * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(hidden_channels * nf_mult_prev, hidden_channels * nf_mult, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.InstanceNorm2d(hidden_channels * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(hidden_channels * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding_size)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

@register_model('fcngen')
def fcngen(input_channels, output_channels, use_tanh=True):
    model = Generator(input_channels, output_channels, use_tanh=use_tanh)
    return model

@register_model('patchgan')
def patchgan(input_channels):
    model = Discriminator(input_channels)
    return model
