import os
import json
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from yaml import load
from tangle.utils import *

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose(
    [
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""
Dataset class
    `PickDataset` for PickNet
    `SepDataset` for SepPositionNet, SepDirectionNet
Author: xinyi
Date: 20220517
"""
class PickDataset(Dataset):
    """
    Output: 
        torch.Size([3, 512, 512]) - 3 channel image
        torch.Size([2, 512, 512]) - 2 channel mask: pick + tangle
    """
    def __init__(self, img_height, img_width, data_folder, data_inds=None, data_type='train'):
        self.img_h = img_height
        self.img_w = img_width
        self.data_type = data_type

        img_folder = os.path.join(data_folder, "images")
        msk_folder = os.path.join(data_folder, "masks")
        lbl_folder = os.path.join(data_folder, "labels")

        self.transform = transforms.Compose([transforms.ToTensor()])
        
        self.images = []
        self.masks = []
        self.labels = []
        if data_type == 'train':
            if data_inds == None:
                num_inds = len(os.listdir(img_folder))
                data_inds = random_inds(num_inds, num_inds)

            for i in data_inds: 
                self.images.append(os.path.join(img_folder, "%06d.png"%i))
                self.masks.append(os.path.join(msk_folder, "%06d.png"%i))
                self.labels.append(np.load(os.path.join(lbl_folder, "%06d.npy"%i))[0])
        
        elif data_type == 'test':

            for f in os.listdir(data_folder):
                if f.split('.')[-1] == 'png': self.images.append(os.path.join(data_folder, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        if self.data_type == 'train':            
                        
            src_img = cv2.imread(self.images[index])
            src_msk = cv2.imread(self.masks[index],0)
            # # check the size
            src_h, src_w = src_msk.shape
            if src_h != self.img_h or src_w != self.img_h:
                src_img = cv2.resize(src_img, (self.img_w, self.img_h))
                src_msk = cv2.resize(src_msk, (self.img_w, self.img_h))

            black = np.zeros((self.img_h,self.img_w), dtype=np.uint8)
            
            if self.labels[index] == 0:
                img = self.transform(src_img)
                msk = self.transform(np.stack([src_msk, black], 2))
            else:
                img = self.transform(src_img)
                msk = self.transform(np.stack([black, src_msk], 2))
            return img, msk

        elif self.data_type == 'test':
            src_img = cv2.imread(self.images[index])
            src_h, src_w,_ = src_img.shape
            if src_h != self.img_h or src_w != self.img_h:
                src_img = cv2.resize(src_img, (self.img_w, self.img_h))
            return self.transform(src_img)

class SepDataset(Dataset):
    """
    Output: 
        torch.Size([5, 512, 512]) - 3 channel image + pull gauss + hold gauss
        torch.Size([2]) - normalized vector for direction, where [0,0] point to right
        Scalar - label: 0/1
    Usage: 
        if 'sep_pos' - 3 channel image --> 2 channel mask
        if 'sep_dir' - 3+2 channel image + direction --> label
    """
    def __init__(self, img_height, img_width, data_folder, net_type, sigma=6, data_type='train', data_inds=None):
        self.img_h = img_height
        self.img_w = img_width
        self.net_type = net_type
        self.sigma = sigma
        self.data_type = data_type

        img_folder = os.path.join(data_folder, "images")
        pos_folder = os.path.join(data_folder, "positions")
        dir_folder = os.path.join(data_folder, "directions")
        lbl_folder = os.path.join(data_folder, "labels")


        self.transform = transforms.Compose([transforms.ToTensor()])
        self.images, self.positions, self.directions, self.labels = [], [], [], []

        if data_type == 'train':
            if data_inds == None:
                num_inds = len(os.listdir(img_folder))
                data_inds = random_inds(num_inds, num_inds)
            for i in data_inds:    
                self.images.append(os.path.join(img_folder, "%06d.png"%i))
                self.positions.append(os.path.join(pos_folder, "%06d.npy"%i))
                self.directions.append(os.path.join(dir_folder, "%06d.npy"%i))
                self.labels.append(os.path.join(lbl_folder, "%06d.npy"%i))
        
        elif data_type == 'test':
            
            for f in os.listdir(data_folder):
                if f.split('.')[-1] == 'png': self.images.append(os.path.join(data_folder, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        if self.data_type == 'train':

            img = cv2.resize(cv2.imread(self.images[index]), (self.img_w, self.img_h))
            p = np.load(self.positions[index]) # pull, hold
            p_no_hold = np.array([np.load(self.positions[index])[0]]) # pull
            q = np.load(self.directions[index])
            l = np.load(self.labels[index])

            img = self.transform(img)
            heatmap = gauss_2d_batch(self.img_h, self.img_w, self.sigma, p)
            heatmap_no_hold = gauss_2d_batch(self.img_h, self.img_w, self.sigma, p_no_hold)

            all_img =  torch.cat((img, heatmap), 0)
            
            all_img_no_hold =  torch.cat((img, heatmap_no_hold), 0)
            q = torch.from_numpy(q)
            l = torch.from_numpy(l)[0]

            if 'pos' in self.net_type: return img, heatmap
            elif 'dir' in self.net_type: return all_img, q, l

        elif self.data_type == 'test':
            img = cv2.resize(cv2.imread(self.images[index]), (self.img_w, self.img_h))
            img = self.transform(img)
            return img
        
            # return all_img_no_hold, q, l


if __name__ == "__main__":

    data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\HoldAndPullDirectionDataAll"
    inds = random_inds(10,50000)
    train_dataset = SepDataset(512, 512, data_folder, net_type='sep_pos', data_inds=inds)

    # for i in range(len(train_dataset)):
    #     img, direction, lbl = train_dataset[i]
    #     print(img.shape, direction.shape, lbl.shape)
    #     direction = direction.detach().cpu().numpy()

    #     print(np.round(vector2direction(direction),1), lbl)

    for i in range(len(train_dataset)):
        img, lbl = train_dataset[i]
        print(img.shape, lbl.shape)

    

    #     depth = visualize_tensor(img[0:3])
    #     pull = visualize_tensor(img[3])
    #     # hold = visualize_tensor(img[4])
    #     plt.imshow(depth)
    #     # plt.imshow(hold, alpha=0.5)
    #     plt.imshow(pull, alpha=0.5)
        # plt.show()
    
    # data_folder = "D:\\datasets\\holdandpull_test"
    # test_dataset = PickDataset(512, 512, data_folder, data_type='test')
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    # print(test_dataset.images)
    # for s in test_loader:
    #     print(s.shape)