"""
Model class
Author: xinyi
Date: 20220517
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tangle.model_parts import resnet50, resnet101,resnet152
from tangle.model_parts import Bridge, UpBlock, MLP, Up, Down, Conv
from torch.utils.data import Dataset, DataLoader
from tangle.utils import gauss_2d_batch, random_inds
from torchvision import transforms
# class SepNetD(nn.Module):
#     """
#     Input: 
#         torch.size([B,5,H,W]) - image + pull map + hold map
#         torch.size([B,2]) - direction vector [x,y]
#     Output: scalar - 0/1
#     Usage: backbone = 'conv' or 'resnet'
#     """
#     def __init__(self,  in_channels=5, backbone='conv'):
#         super().__init__()
#         image_feature_dim = 256
#         action_feature_dim = 128
#         output_dim = 1
#         multi_output_dim = 16
#         self.backbone = backbone
#         self.action_encoder = MLP(2, action_feature_dim, [action_feature_dim, action_feature_dim])
#         self.decoder = MLP(image_feature_dim + action_feature_dim, 2 * output_dim, [1024, 1024, 1024]) # 2 classes
#         self.multi_action_decoder = MLP(image_feature_dim, output_dim, [1024, 1024, 1024]) # 16 classes
        
#         if backbone == 'conv':
#             self.image_encoder_1 = Conv(in_channels, 32)
#             self.image_encoder_2 = Down(32, 64)
#             self.image_encoder_3 = Down(64, 128)
#             self.image_encoder_4 = Down(128, 256)
#             self.image_encoder_5 = Down(256, 512)
#             self.image_encoder_6 = Down(512, 512)
#             self.image_encoder_7 = Down(512, 512)
#             self.image_feature_extractor = MLP(512*8*8, image_feature_dim, [image_feature_dim])

#         if backbone =='resnet':
#             self.resnet = resnet50(pretrained=True)
#             self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
#             modules = list(self.resnet.children())[:-1]      # delete the last fc layer.
#             # # modules.append(nn.Dropout(0.5))
#             self.resnet = nn.Sequential(*modules)
#             self.image_feature_extractor = MLP(2048*2*2, image_feature_dim, [image_feature_dim])
#             # self.linear = nn.Linear(2048, out_features=1)
    
#     def forward(self, observation):
#         """
#         Input: torch.Size([1,5,512,512])
#         Output: torch.Size([1, 16])
#         """
        
#         x0, directions = observation
#         # # print("================= image ==================")
#         if self.backbone == 'conv':
#             x1 = self.image_encoder_1(x0)
#             x2 = self.image_encoder_2(x1)
#             x3 = self.image_encoder_3(x2)
#             x4 = self.image_encoder_4(x3)
#             x5 = self.image_encoder_5(x4)
#             x6 = self.image_encoder_6(x5)
#             x7 = self.image_encoder_7(x6)
#             embedding = x7.reshape([x7.size(0), -1])
#             feature = self.image_feature_extractor(embedding)
        
#         if self.backbone == 'resnet':
#             x1 = self.resnet(x0)
#             embedding = x1
#             feature = self.image_feature_extractor(embedding)
#             print("output ", x1.shape) 
#         direction_features = self.action_encoder(directions)
#         # direction_features = self.action_encoder(directions)

#         # output = None
#         # feature_input = torch.cat([feature, direction_features], dim=1)
#         # output = self.decoder(feature_input)
        
#         # return output
#         # print(feature.shape) 
#         # output = self.decoder(feature)
#         print(feature.shape)
class SepDDataset(Dataset):
    def __init__(self, img_h, img_w, folder, sigma=6, data_inds=None):
        self.img_h = img_h
        self.img_w = img_w
        self.sigma = sigma
        images_folder = os.path.join(folder, "images")
        positions_path = os.path.join(folder, "positions.npy")
        labels_path = os.path.join(folder, "labels.npy")
        data_num = len(os.listdir(images_folder))
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        # self.positions, self.directions, self.labels = np.array([]),np.array([]) ,np.array([]) ,np.array([])  
        self.images = []
        self.positions, self.labels = [], [] 

        positions = np.load(positions_path)
        labels = np.load(labels_path)
        for i in data_inds: 
            self.images.append(os.path.join(images_folder, '%06d.png'%i))
            self.positions.append(positions[i])
            self.labels.append(labels[i])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = cv2.resize(cv2.imread(self.images[index]), (self.img_w, self.img_h))
        p = self.positions[index] # pull, hold
        heatmap = gauss_2d_batch(self.img_h, self.img_w, self.sigma, p)
        img = self.transform(img)
        
        cat_img = torch.cat((img, heatmap))
        l = torch.tensor(self.labels[index])
        return cat_img, l
        
class SepNetD(nn.Module):
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
        output_dim = 16
        self.backbone = backbone
        self.action_encoder = MLP(2, action_feature_dim, [action_feature_dim, action_feature_dim])
        # self.decoder = MLP(image_feature_dim + action_feature_dim, 2 * output_dim, [1024, 1024, 1024]) # 2 classes
        self.decoder = MLP(image_feature_dim, output_dim, [1024, 1024, 1024]) # 18 classes
        
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
    
    def forward(self, observation):
        """
        Input: torch.Size([1,5,512,512]), torch.Size([1, 2])
        Output: torch.Size([1, 2])
        """
        x0 = observation
        # # print("================= image ==================")
        if self.backbone == 'conv':
            x1 = self.image_encoder_1(x0)
            x2 = self.image_encoder_2(x1)
            x3 = self.image_encoder_3(x2)
            x4 = self.image_encoder_4(x3)
            x5 = self.image_encoder_5(x4)
            x6 = self.image_encoder_6(x5)
            x7 = self.image_encoder_7(x6)
            embedding = x7.reshape([x7.size(0), -1])
            feature = self.image_feature_extractor(embedding)
        
        if self.backbone == 'resnet':
            x1 = self.resnet(x0)
            embedding = x1
            feature = self.image_feature_extractor(embedding)
            
        output = self.decoder(feature)
        
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
    model = SepNetD(in_channels=5, backbone='resnet')
    # out = model.forward(inp_img5, inp_direction)
    out = model.forward(inp_img5)
    print(inp_img5.shape, out.shape)
    
    # data_inds = random_inds(10,100)
    # data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\sepnet\\u"
    # dataset = SepDDataset(512, 512, data_folder, data_inds=data_inds)
    # from torchviz import make_dot
    # dot = make_dot(out)
    # dot.format = 'png'
    # dot.render('torchviz-sample')
    # # print(f"SepDirectionNet: ", out.shape)
    # checkpoint = torch.load(model_ckpt)
    # model.load_state_dict(torch.load(model_ckpt))
