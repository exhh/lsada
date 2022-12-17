import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import Indexflow, to_device
from .models import register_model
import numpy as np

__all__ = ['MicroNet', 'micronet']

def passthrough(x, **kwargs):
    return x

def convAct(nchan):
    return nn.ELU(inplace=True)

class ConvBN(nn.Module):
    def __init__(self, nchan, inChans=None):
        super(ConvBN, self).__init__()
        if inChans is None:
            inChans = nchan
        self.act = convAct(nchan)
        self.conv = nn.Conv2d(inChans, nchan, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(nchan)

    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))
        return out

def _make_nConv(nchan, depth):
    layers = []
    if depth >=0:
        for _ in range(depth):
            layers.append(ConvBN(nchan))
        return nn.Sequential(*layers)
    else:
        return passthrough

class InputTransition(nn.Module):
    def __init__(self,inputChans, outChans):
        self.outChans = outChans
        self.inputChans = inputChans
        super(InputTransition, self).__init__()
        self.conv = nn.Conv2d(inputChans, outChans, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(outChans)
        self.relu = convAct(outChans)

    def forward(self, x):
        out = self.norm(self.conv(x))
        if self.inputChans == 1:
            x_aug = torch.cat([x]*self.outChans, 0)
            out = self.relu(torch.add(out, x_aug))
        else:
            out = self.relu(out)
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1, stride=2)
        self.norm1 = nn.InstanceNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = convAct(outChans)
        self.relu2 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout2d()

        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.relu1(self.norm1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(out+down)
        return out

def match_tensor(out, refer_shape):
    skiprow,skipcol = refer_shape
    row, col = out.size()[2], out.size()[3]
    if skipcol >= col:
        pad_col = skipcol - col
        left_pad_col  = pad_col // 2
        right_pad_col = pad_col - left_pad_col
        out = F.pad(out, (left_pad_col, right_pad_col, 0,0))
    else:
        crop_col = col - skipcol
        left_crop_col  = crop_col // 2
        right_col = left_crop_col + skipcol
        out = out[:,:,:, left_crop_col:right_col]

    if skiprow >= row:
        pad_row = skiprow - row
        left_pad_row  = pad_row // 2
        right_pad_row = pad_row - left_pad_row
        out = F.pad(out, (0,0, left_pad_row,right_pad_row))
    else:
        crop_row = row - skiprow
        left_crop_row  = crop_row // 2
        right_row = left_crop_row + skiprow
        out = out[:,:,left_crop_row:right_row, :]
    return out


class UpConcat(nn.Module):
    def __init__(self, inChans, hidChans, outChans, nConvs, dropout=False, stride=2):
        super(UpConcat, self).__init__()
        self.up_samp = nn.Upsample(scale_factor=stride, mode='bilinear')
        self.up_conv = nn.Conv2d(inChans, hidChans, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(hidChans)
        self.do1 = passthrough
        self.do2 = nn.Dropout2d()
        self.relu1 = convAct(hidChans)
        self.relu2 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout2d()
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.norm1(self.up_conv(self.up_samp(out))))
        out = match_tensor(out, skipxdo.size()[2:])

        xcat = torch.cat([out, skipxdo], 1)
        out  = self.ops(xcat)
        out  = self.relu2(out + xcat)
        return out

class UpConv(nn.Module):
    def __init__(self, inChans, outChans, nConvs,dropout=False, stride = 2):
        super(UpConv, self).__init__()
        self.up_samp = nn.Upsample(scale_factor=stride, mode='bilinear')
        self.up_conv = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout2d()

    def forward(self, x, dest_size):
        out = self.do1(x)
        out = self.relu1(self.norm1(self.up_conv(self.up_samp(out))))
        out = match_tensor(out, dest_size)
        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans,outChans=1,hidChans=2):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(inChans, hidChans, kernel_size=5, padding=2)
        self.norm1 = nn.InstanceNorm2d(hidChans)
        self.relu1 = convAct( hidChans)
        self.conv2 = nn.Conv2d(hidChans, outChans, kernel_size=1)

    def forward(self, x):
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.conv2(out)
        return out

class MicroNet(nn.Module):
    def __init__(self, input_channels = 3, output_channels = 1):
        super(MicroNet, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.in_tr    = InputTransition(input_channels, 32)
        self.down_tr1 = DownTransition(32, 64, 1)
        self.down_tr2 = DownTransition(64, 128, 2)
        self.down_tr3 = DownTransition(128, 256, 2,dropout=True)
        self.down_tr4 = DownTransition(256, 256, 2,dropout=True)

        self.up_tr4 = UpConcat(256, 256, 512, 2,  dropout=True)
        self.up_tr3 = UpConcat(512, 128, 256, 2, dropout=True)
        self.up_tr2 = UpConcat(256, 64, 128, 1)
        self.up_tr1 = UpConcat(128, 32, 64, 1)

        self.up3 = UpConv(512, 64, 2, stride = 8)
        self.up2 = UpConv(256, 64, 2, stride = 4)
        self.out_tr = OutputTransition(64*3, output_channels, 32)

    def forward(self, x):
        x = to_device(x,self.device_id)
        out_intr = self.in_tr(x)
        out_dtr1 = self.down_tr1(out_intr)
        out_dtr2 = self.down_tr2(out_dtr1)
        out_dtr3 = self.down_tr3(out_dtr2)
        out_dtr4 = self.down_tr4(out_dtr3)

        out_uptr4 = self.up_tr4(out_dtr4, out_dtr3)
        out_uptr3 = self.up_tr3(out_uptr4, out_dtr2)
        out_uptr2 = self.up_tr2(out_uptr3, out_dtr1)
        out_uptr1 = self.up_tr1(out_uptr2, out_intr)

        out_up3 = self.up3(out_uptr4, x.size()[2:])
        out_up2 = self.up2(out_uptr3, x.size()[2:])
        out_cat = torch.cat([out_uptr1, out_up3, out_up2], 1)
        out = self.out_tr(out_cat)
        return out

    def predict(self, batch_data, batch_size=None):
        self.eval()
        total_num = batch_data.shape[0]
        if batch_size is None or batch_size >= total_num:
            x = to_device(batch_data, self.device_id, False).float()
            det = self.forward(x)
            return det
        else:
            results = []
            for ind in Indexflow(total_num, batch_size, False):
                data = batch_data[ind]
                data = to_device(data, self.device_id, False).float()
                det = self.forward(data)
                results.append(det)
            return torch.cat(results,dim=0)

@register_model('micronet')
def micronet(input_channels=3, output_channels=1, **kwargs):
    model = MicroNet(input_channels=input_channels, output_channels=output_channels)
    return model
