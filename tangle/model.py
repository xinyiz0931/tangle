"""
Model class
Author: xinyi
Date: 20220517
"""
import os
import torch
import torch.nn as nn
import torchvision
from tangle.model_parts import resnet50, resnet101,resnet152
from tangle.model_parts import Bridge, UpBlock, MLP, Up, Down, Conv

class PickNet(nn.Module):
    """
    Input: torch.size([B,3,H,W])
    Output: torch.size([B,2,H,W]) - pick/sep affordance maps
    Usage: 
        model_type = 'unet' or 'fcn'
    """
    DEPTH = 6
    def __init__(self, model_type='unet', out_channels=2):
        super().__init__()
        self.model_type = model_type
        self.out_channels = out_channels
        if model_type == 'unet': 
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
        if model_type == 'fcn':
            self.fcn = torch.hub.load("pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=False)

    def forward(self, x, with_output_feature_map=False):
        if self.model_type == 'unet': 
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

        if self.model_type == 'fcn':
            return self.fcn(x)
            # x = self.fcn(x)['out']
            # x = x[:,:self.out_channels, :, :]
            # return x

class SepPositionNet(nn.Module):
    """
    Input: torch.size([B,3,H,W]) - image
    Output: torch.size([B,2,H,W]) - pull map + hold map
    Usage: model_type = 'unet' or 'fcn'
    """
    def __init__(self, out_channels, img_height=512, img_width=512):
        super(SepPositionNet, self).__init__()
        self.out_channels = out_channels
        self.img_height = img_height
        self.img_width = img_width
        resnet = resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
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

class SepDirectionNet(nn.Module):
    """
    Input: 
        torch.size([B,5,H,W]) - image + pull map + hold map
        torch.size([B,2]) - direction vector [x,y]
    Output: scalar - 0/1
    Usage: backbone = 'conv' or 'resnet'
    """
    def __init__(self,  in_channels=5, backbone='conv'):
        super().__init__()
        image_feature_dim = 256
        action_feature_dim = 128
        output_dim = 1
        self.backbone = backbone
        self.action_encoder = MLP(2, action_feature_dim, [action_feature_dim, action_feature_dim])
        self.decoder = MLP(image_feature_dim + action_feature_dim, 2 * output_dim, [1024, 1024, 1024]) # 2 classes
        
        if backbone == 'conv':
            self.image_encoder_1 = Conv(in_channels, 32)
            self.image_encoder_2 = Down(32, 64)
            self.image_encoder_3 = Down(64, 128)
            self.image_encoder_4 = Down(128, 256)
            self.image_encoder_5 = Down(256, 512)
            self.image_encoder_6 = Down(512, 512)
            self.image_encoder_7 = Down(512, 512)
            self.image_feature_extractor = MLP(512*8*8, image_feature_dim, [image_feature_dim])
        if backbone =='resnet':
            self.resnet = resnet101(pretrained=True)
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
            modules = list(self.resnet.children())[:-1]      # delete the last fc layer.
            # modules.append(nn.Dropout(0.5))
            self.resnet = nn.Sequential(*modules)
            self.image_feature_extractor = MLP(2048*2*2, image_feature_dim, [image_feature_dim])
            # self.linear = nn.Linear(2048, out_features=1)
    
    def forward(self, observation, directions):
        """
        Input: torch.Size([1,5,512,512]), torch.Size([1, 2])
        Output: torch.Size([1, 2])
        """
        x0 = observation
        # # print("================= image ==================")
        if self.backbone == 'conv':
            x1 = self.image_encoder_1(x0)
            # print(f"x0 -> x1: {x0.shape} -> {x1.shape}")
            x2 = self.image_encoder_2(x1)
            # print(f"x1 -> x2: {x1.shape} -> {x2.shape}")
            x3 = self.image_encoder_3(x2)
            # print(f"x2 -> x3: {x2.shape} -> {x3.shape}")
            x4 = self.image_encoder_4(x3)
            # print(f"x3 -> x4: {x3.shape} -> {x4.shape}")
            x5 = self.image_encoder_5(x4)
            # print(f"x4 -> x5: {x4.shape} -> {x5.shape}")
            x6 = self.image_encoder_6(x5)
            # print(f"x5 -> x6: {x5.shape} -> {x6.shape}")
            x7 = self.image_encoder_7(x6)
            # print(f"x6 -> x7: {x6.shape} -> {x7.shape}")
            embedding = x7.reshape([x7.size(0), -1])
            # print(f"x7 -> image encoder output: {x7.shape} -> {embedding.shape}")
            feature = self.image_feature_extractor(embedding)
            # print(f"image decoder input -> output: {embedding.shape} -> {feature.shape}")
        
        if self.backbone == 'resnet':
            x1 = self.resnet(x0)
            # print(f"x0 -> x1: {x0.shape} -> {x1.shape}")
            embedding = x1
            # print(f"x1 -> image encoder output: {x1.shape} -> {embedding.shape}")
            feature = self.image_feature_extractor(embedding)
            # print(f"image decoder input -> output: {embedding.shape} -> {feature.shape}")
            
        # # print("================= action ==================")
        direction_features = self.action_encoder(directions)

        output = None
        feature_input = torch.cat([feature, direction_features], dim=1)
        output = self.decoder(feature_input)
        
        return output

if __name__ == '__main__':
    # root_dir = "C:\\Users\\xinyi\\Documents"
    # model_ckpt = os.path.join(root_dir, "Checkpoints", "try_SR_", "model_epoch_7.pth")
    batch_size = 1
    inp_img3 = torch.rand((batch_size, 3, 512, 512))
    inp_img4 = torch.rand((batch_size, 4, 512, 512))
    inp_img5 = torch.rand((batch_size, 5, 512, 512))
    inp_direction = torch.rand((batch_size,2))

    # 1 PickNet
    # model = PickNet(model_type='fcn', out_channels=2)
    # model = PickNet(model_type='unet', out_channels=2)
    # # model = torch.hub.load("pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=True)
    # # model.load_state_dict(torch.load(model_ckpt))
    # out = model.forward(inp_img3)
    # print(f"PickNet: ", out.shape)

    # 2.1 SepPositionNet
    #model = SepPositionNet(out_channels=2)
    #out = model.forward(inp_img3)
    ## print(f"SepPositionNet: ", out.shape)
    

    # 2.2 SepDirectionnet
    model = SepDirectionNet(in_channels=4, backbone='resnet')
    out = model.forward(inp_img4, inp_direction)
    print(inp_img4.shape, inp_direction.shape, out.shape)
    from torchviz import make_dot
    dot = make_dot(out)
    dot.format = 'png'
    dot.render('torchviz-sample')
    # # print(f"SepDirectionNet: ", out.shape)
    # checkpoint = torch.load(model_ckpt)
    # model.load_state_dict(torch.load(model_ckpt))
