import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import filters

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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
# UNet class


def rgb_to_grayscale(tensor):
    # Assuming tensor shape is (batch, channels, height, width)
    # Weighted sum for conversion to grayscale
    grayscale_tensor = 0.299 * tensor[:, 0, :, :] + 0.587 * tensor[:, 1, :, :] + 0.114 * tensor[:, 2, :, :]
    
    # Add a channel dimension to the grayscale tensor
    grayscale_tensor = grayscale_tensor.unsqueeze(1)
    
    return grayscale_tensor
  
def apply_otsu_threshold(grayscale_tensor):
    # Assuming grayscale_tensor shape is (batch, 1, height, width)

    # Convert PyTorch tensor to NumPy array
    grayscale_np = grayscale_tensor.cpu().numpy()

    # Initialize an empty tensor for the thresholded result
    thresholded_tensor = torch.zeros_like(grayscale_tensor)

    # Loop through each batch
    for i in range(grayscale_tensor.shape[0]):
        # Apply Otsu's method using skimage.filters.threshold_otsu
        threshold_value = filters.threshold_otsu(grayscale_np[i, 0, :, :])

        # Apply thresholding to create a binary mask
        binary_mask = (grayscale_np[i, 0, :, :] > threshold_value).astype(float)

        # Assign the binary mask to the result tensor
        thresholded_tensor[i, 0, :, :] = torch.tensor(binary_mask)

    return thresholded_tensor

def broadcast_to_rgb(grayscale_tensor):
    # Assuming grayscale_tensor shape is (batch, 1, height, width)

    rgb_tensor = grayscale_tensor.expand(-1, 3, -1, -1)

    return rgb_tensor

def apply_otsu(img):
  img = rgb_to_grayscale(img)
  img = apply_otsu_threshold(img)
  img = broadcast_to_rgb(img)
  return img

  
class UNet(nn.Module):
    def __init__(
            self, input_channels=3, nclasses=1, features=[64, 128, 256, 512],
    ):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        in_channels = input_channels
        out_channels = nclasses
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
          x = apply_otsu(x)
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)