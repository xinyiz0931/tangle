"""
Model class
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
    UNet model from https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder.git
    """
    DEPTH = 6
    def __init__(self, model_type="unet", out_channels=2):
        super().__init__()
        self.model_type = model_type
        self.out_channels = out_channels
        if model_type == "unet": 
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
        if model_type == "fcn":
            self.fcn = torch.hub.load("pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=False)

    def forward(self, x, with_output_feature_map=False):
        if self.model_type == "unet": 
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

        if self.model_type == "fcn":
            return self.fcn(x)
            # x = self.fcn(x)["out"]
            # x = x[:,:self.out_channels, :, :]
            # return x

class SepNet(nn.Module):
    """
    Input: torch.size([B,3,H,W]) - image
    Output: torch.size([B,2,H,W]) - pull map + hold map
    Usage: model_type = "unet" or "fcn"
    """
    def __init__(self, out_channels, in_channels=3, backbone=""):
        super(SepNet, self).__init__()
        self.out_channels = out_channels
        from tangle.model_parts import resnet34, resnet18
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

# --------------------------------------------------------------------------

class SepNetD(nn.Module):
    """
    Input: 
        torch.size([B,5,H,W]) - image + pull map + hold map
        torch.size([B,2]) - direction vector [x,y]
    Output: scalar - 0/1
    Usage: backbone = "conv" or "resnet"
    """
    def __init__(self,  in_channels=5, backbone="conv"):
        super().__init__()
        image_feature_dim = 256
        action_feature_dim = 128
        output_dim = 1
        self.backbone = backbone
        self.action_encoder = MLP(2, action_feature_dim, [action_feature_dim, action_feature_dim])
        
        if backbone == "conv":
            self.image_encoder_1 = Conv(in_channels, 32)
            self.image_encoder_2 = Down(32, 64)
            self.image_encoder_3 = Down(64, 128)
            self.image_encoder_4 = Down(128, 256)
            self.image_encoder_5 = Down(256, 512)
            self.image_encoder_6 = Down(512, 512)
            self.image_encoder_7 = Down(512, 512)
            self.image_feature_extractor = MLP(512*8*8, image_feature_dim, [image_feature_dim])
            # self.image_feature_extractor = MLP(512*7*7, image_feature_dim, [image_feature_dim])
            self.decoder = MLP(image_feature_dim + action_feature_dim, 2 * output_dim, [1024, 1024, 1024]) # 2 classes

        elif "resnet" in backbone:

            resnet = torch.hub.load('pytorch/vision:v0.10.0', backbone, pretrained=True)
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
            modules = list(resnet.children())[:-1]      # delete the last fc layer.
            # modules.append(nn.Dropout(0.5))
            self.resnet = nn.Sequential(*modules)
            # self.image_feature_extractor = MLP(2048*2*2, image_feature_dim, [image_feature_dim])
            # self.linear = nn.Linear(2048, out_features=1)
            self.decoder = MLP(2048 + action_feature_dim, 2 * output_dim, [1024, 1024, 1024]) # 2 classes
    
    def forward(self, x):
        """
        Input: torch.Size([1,5,512,512]), torch.Size([1, 2])
        Output: torch.Size([1, 2])
        """
        observation, directions = x
        if self.backbone == "conv":
            x = self.image_encoder_1(observation)
            x = self.image_encoder_2(x)
            x = self.image_encoder_3(x)
            x = self.image_encoder_4(x)
            x = self.image_encoder_5(x)
            x = self.image_encoder_6(x)
            x = self.image_encoder_7(x)
            x = x.reshape([x.size(0), -1])
            image_features = self.image_feature_extractor(x)
        
        if "resnet" in self.backbone:
            x = self.resnet(observation)
            image_features = x.view(x.size(0), -1)

        direction_features = self.action_encoder(directions)
        feature_input = torch.cat([image_features, direction_features], dim=1)
        output = self.decoder(feature_input)
        
        return output

class SepNetD_Multi(nn.Module):
    """
    Input: 
        torch.size([B,5,W,H]) - image + pull map + hold map
    Output:
        torch.size([B,16]) - scores for 16 directions
    Usage: backbone = "conv" or "resnet"
    """
    def __init__(self, in_channels=5, backbone="conv"):
        super().__init__()
        image_feature_dim = 256
        output_dim = 16
        self.backbone = backbone
        self.decoder = MLP(image_feature_dim, output_dim, [1024, 1024, 1024]) # 16 classes

        if backbone == "conv":
            self.image_encoder_1 = Conv(in_channels, 32)
            self.image_encoder_2 = Down(32, 64)
            self.image_encoder_3 = Down(64, 128)
            self.image_encoder_4 = Down(128, 256)
            self.image_encoder_5 = Down(256, 512)
            self.image_encoder_6 = Down(512, 512)
            self.image_encoder_7 = Down(512, 512)
            self.image_feature_extractor = MLP(512*8*8, image_feature_dim, [image_feature_dim])
        
        if backbone == "resnet":
            from torchvision.models import resnet50
            self.resnet = resnet50(pretrained=True)
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, output_dim)
        
    def forward(self, x):
        if self.backbone == "conv":
            x1 = self.image_encoder_1(x)
            x2 = self.image_encoder_2(x1)
            x3 = self.image_encoder_3(x2)
            x4 = self.image_encoder_4(x3)
            x5 = self.image_encoder_5(x4)
            x6 = self.image_encoder_6(x5)
            x7 = self.image_encoder_7(x6)
            embedding = x7.reshape([x7.size(0), -1])
            feature = self.image_feature_extractor(embedding)
            output = self.decoder(feature)
        elif self.backbone == "resnet":
            output = self.resnet(x)
        return output

class SepNetD_AM(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        from tangle.model_parts import resnet34
        resnet = resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
    # root_dir = "C:\\Users\\xinyi\\Documents"
    # model_ckpt = os.path.join(root_dir, "Checkpoints", "try_SR_", "model_epoch_7.pth")
    batch_size = 1
    inp_img3 = torch.rand((batch_size, 3, 512, 512))
    inp_img4 = torch.rand((batch_size, 4, 512, 512))
    inp_img5 = torch.rand((batch_size, 5, 512, 512))
    inp_direction = torch.rand((batch_size,2))

    # ----------------------- PickNet ---------------------------- 
    # model = PickNet(model_type="fcn", out_channels=2)
    model = PickNet(model_type="fcn", out_channels=1)
    # model = torch.hub.load("pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=True)
    # model.load_state_dict(torch.load(model_ckpt))
    out = model.forward(inp_img3)
    print(f"PickNet: ", out["out"][:,:1,:,:].shape)

    # ----------------------- SepNet-P ---------------------------- 
    model = SepNet(out_channels=2)
    out = model.forward(inp_img3)
    print("SepNet-P: ", inp_img3.shape, "=>", out.shape)

    # ----------------------- SepNet-D ---------------------------- 
    # ckpt = "C:\\Users\\xinyi\\Documents\\Checkpoint\\try_new_res\\model_epoch_12.pth"
    # ckpt = "C:\\Users\\xinyi\\Documents\\Checkpoint\\try_SR\\model_epoch_99.pth"
    ckpt = "C:\\Users\\xinyi\\Documents\\Checkpoint\\try_sepnet_vector_eight\\model_epoch_10.pth"
    model = SepNetD(in_channels=5, backbone="conv")
    out = model.forward((inp_img5, inp_direction))
    model.load_state_dict(torch.load(ckpt))
    print("SepNet-D: ", inp_img5.shape, inp_direction.shape, "=>", out.shape)

    # ----------------------- SepNet-D Action-spatial map ---------------------------- 
    model = SepNet(in_channels=4, out_channels=1).cuda()
    inp_img4 = inp_img4.cuda()
    
    out = model.forward(inp_img4)
    print("SepNet-D (AM): ", out.shape)
    # ----------------------- SepNet-D Multi ---------------------------- 
    
    # model = SepNetD_Multi(in_channels=5, backbone="conv")
    # model.load_state_dict(torch.load(ckpt))

    # out = model.forward(inp_img5)
    # print(inp_img5.shape, "=>", out.shape)
   