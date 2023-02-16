"""
Author: xinyi
Date: 20220517
"""
import os
import torch
import torch.nn as nn
import torchvision
from tangle.model_parts import Bridge, UpBlock, MLP, Up, Down, Conv
import warnings
warnings.filterwarnings("ignore")
class PickNet(nn.Module):
    """
    Input: torch.size([B,3,H,W])
    Output: torch.size([B,2,H,W]) - pick/sep affordance maps
    Usage: 
        model_type = "unet" or "fcn"
    UNet model from https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder
    """
    DEPTH = 6
    def __init__(self, out_channels=2):
        super().__init__()
        self.out_channels = out_channels
        # resnet = torchvision.models.resnet.resnet101(pretrained=True)
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlock(2048, 1024))
        up_blocks.append(UpBlock(1024, 512))
        up_blocks.append(UpBlock(512, 256))
        up_blocks.append(UpBlock(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlock(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (PickNet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{PickNet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

class PullNet(nn.Module):
    """
    Input: torch.size([B,in_channels,H,W]) - image
    Output: torch.size([B,out_channels,H,W]) - pull map + hold map
    model from https://github.com/jenngrannen/hulk-keypoints
    """
    def __init__(self, out_channels, in_channels=3):
        super(PullNet, self).__init__()
        self.out_channels = out_channels
        from tangle.model_parts import resnet50, resnet34, resnet18
        resnet = resnet18(fully_conv=True,
                          pretrained=True,
                          output_stride=8,
                          remove_avg_pool_layer=True)
        
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        resnet.fc = nn.Conv2d(resnet.inplanes, 1000, 1)
        self.resnet = resnet
        self._normal_initialization(self.resnet.fc)
        self.sigmoid = torch.nn.Sigmoid()

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]
        x = self.resnet(x)
        output = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        heatmaps = self.sigmoid(output[:,:self.out_channels, :, :])
        return heatmaps

if __name__ == "__main__":
    batch_size = 1
    
    img = torch.rand((batch_size, 3, 512, 512))

    # ----------------------- PickNet ---------------------------- 
    model = PickNet(out_channels=2)
    out = model.forward(img)
    print(f"PickNet: {img.shape} => {out.shape}")

    # ----------------------- PullNet ---------------------------- 
    model = PullNet(out_channels=1)
    out = model.forward(img)
    print(f"PullNet: {img.shape} => {out.shape}")
