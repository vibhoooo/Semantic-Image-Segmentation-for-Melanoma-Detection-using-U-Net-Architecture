"""
UNet
The main UNet model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility Functions
''' when filter kernel= 3x3, padding=1 makes in&out matrix same size'''


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def down_pooling():
    return nn.MaxPool2d(2)


def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# UNet class

class UNet(nn.Module):
    def __init__(self, input_channels, nclasses):
        super().__init__()
        # go down
        self.conv1 = conv_bn_relu(input_channels, 64)
        self.conv2 = conv_bn_relu(64, 128)
        self.conv3 = conv_bn_relu(128, 256)
        self.conv4 = conv_bn_relu(256, 512)
        self.conv5 = conv_bn_relu(512, 1024)
        self.down_pooling = nn.MaxPool2d(2)

        # go up
        self.up_pool6 = up_pooling(1024, 512)
        self.conv6 = conv_bn_relu(1024, 512)
        self.up_pool7 = up_pooling(512, 256)
        self.conv7 = conv_bn_relu(512, 256)
        self.up_pool8 = up_pooling(256, 128)
        self.conv8 = conv_bn_relu(256, 128)
        self.up_pool9 = up_pooling(128, 64)
        self.conv9 = conv_bn_relu(128, 64)

        self.conv10 = nn.Conv2d(64, nclasses, 1)

        # test weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # normalize input data
        x = x / 255.
        # go down
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        # go up
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = F.sigmoid(output)

        return output


class UNet2(nn.Module):
    def __init__(self, input_channels, nclasses):
        super().__init__()
        # go down
        self.conv1 = conv_bn_relu(input_channels, 16)
        self.conv2 = conv_bn_relu(16, 32)
        self.conv3 = conv_bn_relu(32, 64)
        self.conv4 = conv_bn_relu(64, 128)
        self.conv5 = conv_bn_relu(128, 256)
        self.down_pooling = nn.MaxPool2d(2)

        # go up
        self.up_pool6 = up_pooling(256, 128)
        self.conv6 = conv_bn_relu(256, 128)
        self.up_pool7 = up_pooling(128, 64)
        self.conv7 = conv_bn_relu(128, 64)
        self.up_pool8 = up_pooling(64, 32)
        self.conv8 = conv_bn_relu(64, 32)
        self.up_pool9 = up_pooling(32, 16)
        self.conv9 = conv_bn_relu(32, 16)

        self.conv10 = nn.Conv2d(16, nclasses, 1)

        # test weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # normalize input data
        x = x / 255.
        # go down
        x1 = self.conv1(x)
        print("X1 shape: ", x1.size())
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        print("X2 Shape: ", x2.size())
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        print("X3 Shape: ", x3.size())
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        print("X4 Shape: ", x4.size())
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)
        print("X5 Shape: ", x5.size())

        # go up
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = F.sigmoid(output)

        return output


"""
Code for Attention UNET
"""


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_Net(nn.Module):
    def __init__(self, input_channels=3, nclasses=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=input_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, nclasses, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


"""
Code for R2_UNET
"""


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class R2U_Net(nn.Module):
    def __init__(self, input_channels=3, nclasses=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=input_channels, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, nclasses, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


"""
STN UNET
"""


class STN_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        xs = self.localization1(x)
        xs = F.avg_pool2d(xs, kernel_size=(61, 61))
        xs = xs.view(1, 64)
        xs = torch.squeeze(xs)
        xs = self.fc(xs)
        theta = xs.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class STN_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        xs = self.localization2(x)

        xs = F.avg_pool2d(xs, kernel_size=(62, 62))

        xs.view(1, 64)
        xs = torch.squeeze(xs)

        xs = self.fc(xs)

        theta = xs.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class STN_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        xs = F.avg_pool2d(x, kernel_size=(64, 64))
        xs = torch.squeeze(xs)
        xs = xs.view(1, 64)
        xs = self.fc(xs)
        theta = xs.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class STN_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        xs = F.avg_pool2d(x, kernel_size=(32, 32))
        xs = torch.squeeze(xs)
        xs = xs.view(1, 128)
        xs = self.fc(xs)
        print(xs.size())
        theta = xs.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class STN_UNet2(nn.Module):
    def __init__(self, input_channels=3, nclasses=1):
        super().__init__()
        # go down
        self.conv1 = conv_bn_relu(input_channels, 16)
        self.conv2 = conv_bn_relu(16, 32)
        self.conv3 = conv_bn_relu(32, 64)
        self.conv4 = conv_bn_relu(64, 128)
        self.conv5 = conv_bn_relu(128, 256)
        self.down_pooling = nn.MaxPool2d(2)

        # go up
        self.up_pool6 = up_pooling(256, 128)
        self.conv6 = conv_bn_relu(256, 128)
        self.up_pool7 = up_pooling(128, 64)
        self.conv7 = conv_bn_relu(128, 64)
        self.up_pool8 = up_pooling(64, 32)
        self.conv8 = conv_bn_relu(64, 32)
        self.up_pool9 = up_pooling(32, 16)
        self.conv9 = conv_bn_relu(32, 16)

        self.conv10 = nn.Conv2d(16, nclasses, 1)

        # STN Layers:
        self.stn1 = STN_1()
        self.stn2 = STN_2()
        self.stn3 = STN_3()
        self.stn4 = STN_4()

        # test weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # normalize input data
        x = x / 255.
        # go down
        x1 = self.conv1(x)
        print("X1 shape: ", x1.size())
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        print("X2 Shape: ", x2.size())
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        print("X3 Shape: ", x3.size())
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        print("X4 Shape: ", x4.size())
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)
        print("X5 Shape: ", x5.size())

        # go up
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, self.stn4(x4)], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, self.stn3(x3)], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, self.stn2(x2)], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, self.stn1(x1)], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = F.sigmoid(output)

        return output

# class STN_2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.localization2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=5),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(True),
#             nn.Conv2d(64, 192, kernel_size=5),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(True),
#             nn.Conv2d(192, 256, kernel_size=5),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(True)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(256 * 12 * 12, 12000),
#             nn.ReLU(True),
#             nn.Linear(12000, 4096),
#             nn.ReLU(True),
#             nn.Linear(4096, 6)
#         )
#
#     def forward(self, x):
#         xs = self.localization2(x)
#         xs = xs.view(-1, 256 * 12 * 12)
#         xs = self.fc(xs)
#         theta = xs.view(-1, 2, 3)
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#         return x
#
#
# class STN_3(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.localization3_1 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=7),
#             nn.MaxPool2d(5, stride=5),
#             nn.ReLU(True)
#         )
#         self.localization3_2 = nn.Sequential(
#             nn.Conv2d(128, 192, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(192 * 3 * 3, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 6)
#         )
#
#     def forward(self, x):
#         xs = self.localization3_1(x)
#         xs = self.localization3_2(xs)
#         xs = xs.view(-1, 192 * 3 * 3)
#         xs = self.fc(xs)
#         theta = xs.view(-1, 2, 3)
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#
#
# class STN_4(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.localization4_1 = nn.Sequential(
#             nn.Conv2d(128, 192, kernel_size=5),
#             nn.MaxPool2d(5, stride=5),
#             nn.ReLU(True)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(192 * 5 * 5, 400),
#             nn.ReLU(True),
#             nn.Linear(400, 6)
#         )
#
#     def forward(self, x):
#         xs = self.localization4_1(x)
#         xs = xs.view(-1, 192 * 5 * 5)
#         xs = self.fc(xs)
#         theta = xs.view(-1, 2, 3)
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
