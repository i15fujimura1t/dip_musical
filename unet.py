import torch
import torch.optim
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down(1, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        self.skip1 = Skip(32, 4)
        self.down4 = Down(32, 64)
        self.skip2 = Skip(64, 4)
        self.downup = DownUp(64, 128)
        self.up1 = Up(132, 128)
        self.up2 = Up(132, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)
        self.last = Last(16, 8)

    def forward(self, x):
        print('a')
        x = self.down1(x)
        x = self.down2(x)
        x3 = self.down3(x)
        x4 = self.down4(x3)
        x = torch.cat((self.downup(x4), self.skip2(x4)), 1)
        x = torch.cat((self.up1(x), self.skip1(x3)), 1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.last(x)
        return x

class Down(nn.Module):
    'Downsampling'

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    'Upsampling'

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, x):
        return self.up(x)

class DownUp(nn.Module):
    'Upsampling then Downsampling'

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downup = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, x):
        return self.downup(x)

class Last(nn.Module):
    'convolute into 1 channel'

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.last = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.last(x)

class Skip(nn.Module):
    'skip connection'

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.skip(x)

def init_weights(m):
    torch.manual_seed(1)
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)
