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

class resNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1,img_width=256):# input img 256X256
        super(resNet, self).__init__() 
        self.out_channel = out_channel
        self.x7conv3to64 = nn.Conv2d(in_channel,64,7,1,3,bias=False)
        self.x1conv64to64 = nn.Conv2d(64,64,1,1,bias=False)
        self.x1conv128to64 = nn.Conv2d(128,64,1,1,bias=False)


        self.x3conv64to64 = nn.Conv2d(64,64,3,1,1,bias=False)
        self.x3conv64to128 = nn.Conv2d(64,128,3,1,1,bias=False)
        self.x3conv128to64 = nn.Conv2d(128,64,3,1,1,bias=False)
        self.x3conv128to128 = nn.Conv2d(128,128,3,1,1,bias=False)
        self.x3conv128to256 = nn.Conv2d(128,256,3,1,1,bias=False)
        self.x3conv256to128 = nn.Conv2d(256,128,3,1,1,bias=False)
        self.x3conv256to256 = nn.Conv2d(256,256,3,1,1,bias=False) 
        self.x3conv256to512 = nn.Conv2d(256,512,3,1,1,bias=False) 
        self.x3conv512to256 = nn.Conv2d(512,256,3,1,1,bias=False)
        self.x3conv512to512 = nn.Conv2d(512,512,3,1,1,bias=False)
        self.x3conv512to1024 = nn.Conv2d(512,1024,3,1,1,bias=False)
        self.x3conv1024to512 = nn.Conv2d(1024,512,3,1,1,bias=False)
        self.x3conv1024to1024 = nn.Conv2d(1024,1024,3,1,1,bias=False)

        self.normal64 = nn.BatchNorm2d(64)
        self.normal128 = nn.BatchNorm2d(128)
        self.normal256 = nn.BatchNorm2d(256)
        self.normal512 = nn.BatchNorm2d(512)
        self.normal1024 = nn.BatchNorm2d(1024)
        self.d1normal1024 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(2)
        self.fc_pool = nn.AvgPool2d(int(img_width/16), stride=1)

        self.linear1024to1024 = nn.Linear(1024, 1024)
        self.linear1024toout = nn.Linear(1024, out_channel)
        self.drop = nn.Dropout(0.1)

        # torch.cat([x, y], dim=1)
        

    
    def forward(self, x):
        # print('0.2', x.shape)
        output = self.x7conv3to64(x)
        # print('0.3', output.shape)
        output = self.pool(output)
        # print('device =', output.device)
        res = self.x3conv64to64(output)
        res = self.x1conv64to64(res)
        # print('2',output.shape, res.shape)
        output = self.normal64(output)
        output = self.relu(self.x3conv64to64(output))
        output = self.relu(self.x3conv64to64(output))
        
        output = torch.cat([output, res], dim=1)
        res = self.x1conv128to64(output)
        output = self.relu(self.x3conv128to64(output))
        output = self.normal64(output)
        output = self.relu(self.x3conv64to64(output))

        output = torch.cat([output, res], dim=1)
        res = self.x1conv128to64(output)
        
        output = self.relu(self.x3conv128to64(output))
        output = self.normal64(output)
        output = self.relu(self.x3conv64to64(output))

        output = torch.cat([output, res], dim=1)  # 1
        output = self.x3conv128to128(output)
        output = self.pool(output)
        res = output
        output = self.normal128(output)
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1) # 2
        output = self.x3conv256to128(output)
        res = output
        # print('2')
        output = self.normal128(output)
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1)# 3
        output = self.x3conv256to128(output)
        res = output
        output = self.normal128(output)
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1)# 4
        output = self.x3conv256to128(output)
        res = output
        output = self.normal128(output)
        output = self.relu(self.x3conv128to128(output))
        output = self.relu(self.x3conv128to128(output))

        output = torch.cat([output, res], dim=1)# 1
        output = self.x3conv256to256(output)
        output = self.pool(output)
        res = output
        # print('3')
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))
        
        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to256(output)
        res = output
        output = self.normal256(output)
        output = self.relu(self.x3conv256to256(output))
        output = self.relu(self.x3conv256to256(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv512to512(output)
        res = output
        output = self.normal512(output)
        output = self.relu(self.x3conv512to512(output))
        output = self.relu(self.x3conv512to512(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv1024to512(output)
        res = output
        output = self.normal512(output)
        output = self.relu(self.x3conv512to512(output))
        output = self.relu(self.x3conv512to512(output))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv1024to512(output)
        res = output
        output = self.normal512(output)
        output = self.relu(self.x3conv512to512(output))
        output = (self.relu(self.x3conv512to512(output)))

        output = torch.cat([output, res], dim=1)
        output = self.x3conv1024to1024(output)
        output = self.pool(output)
        output = self.x3conv1024to1024(output)
        output = self.x3conv1024to1024(output)
        # print('4.4', output.shape)
        output = self.fc_pool(output)
        # print('4.5', output.shape)
        output = output.reshape(output.shape[0], -1)
        # print('4.6', output.shape)
        if output.shape[0] > 1:
            output = self.d1normal1024(output)
        # print('5', output.shape)
        output = self.drop(self.relu(self.linear1024to1024(output)))
        if output.shape[0] > 1:
            output = self.d1normal1024(output)
        output = self.relu(self.linear1024to1024(output))
        output = self.drop(self.relu(self.linear1024to1024(output)))
        if output.shape[0] > 1:
            output = self.d1normal1024(output)
        output = self.linear1024toout(output)
        # print('device =', output.device)
        # print('6', output.shape)
        return output

if __name__ == "__main__":
    x = torch.randn(10, 1, 256, 256)
    res = resNet(1, 1, 256)
    y = res(x)

    print(x.shape)
    print(y.shape)
