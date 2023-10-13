import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # else:
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_plus(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_plus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)     # X 1,0
        x01 = self.up4(x2, x1)
        x3 = self.down2(x2)     # X 2,0
        x11 = self.up3(x3, x2)
        x02 = self.up4(x11, x01)
        x4 = self.down3(x3)     # X 3,0
        x21 = self.up2(x4, x3)
        x12 = self.up3(x21, x11)
        x03 = self.up4(x12, x02)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x21)
        x = self.up3(x, x12)
        x = self.up4(x, x03)
        logits = self.outc(x)
        return logits

class CNN(nn.Module):
    def __init__(self, n_channels, n_classes, img_width=224, img_height=224):
        super(CNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_width = int(img_width / 4)
        self.img_height = int(img_height / 4)
        self.maxpool = nn.MaxPool2d(2)

        self.inc = (DoubleConv(n_channels,8))
        self.down1 = (Down(8, 8))
        self.down2 = (Down(16, 16))
        self.fc1 = nn.Linear(32*self.img_width*self.img_height, int(self.img_width*self.img_height / 6))
        self.fc2 = nn.Linear(int(self.img_width*self.img_height / 6), int(self.img_width*self.img_height / 6))
        self.fc3 = nn.Linear(int(self.img_width*self.img_height / 6), n_classes)
        self.ReLU = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        

        x1 = self.inc(x)
        x2 = self.down1(x1)

        x1 = self.maxpool(x1)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.down2(x2)

        x2 = self.maxpool(x2)
        x3 = torch.cat([x2, x3], dim=1)

        x3 = x3.reshape(x3.shape[0], -1)
        h1 = self.ReLU(self.fc1(x3))
        h1 = self.drop(h1)
        h2 = self.ReLU(self.fc2(h1))
        h2 = self.drop(h2)

        h2 = self.ReLU(self.fc2(h2))
        h2 = self.drop(h2)
        h2 = self.ReLU(self.fc2(h2))
        h2 = self.drop(h2)
        h2 = self.ReLU(self.fc2(h2))
        h2 = self.drop(h2)
        h2 = self.ReLU(self.fc2(h2))
        h2 = self.drop(h2)
        h2 = self.ReLU(self.fc2(h2))
        h2 = self.drop(h2)
        


        h3 = self.ReLU(self.fc3(h2))
        return h3


if __name__ == "__main__":
    x = torch.randn(10, 1, 448, 448)
    cnn = CNN(1, 3, 448, 448)
    y = cnn(x)

    print(x.shape)
    print(y.shape)
