import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
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
        self.dir_list = []
        for i in range(16):
            self.dir_list.append(angle2vector(i*360/16))

        img_folder = os.path.join(data_folder, "images")
        pos_folder = os.path.join(data_folder, "positions")
        crs_folder = os.path.join(data_folder, "heatmaps")
        dir_folder = os.path.join(data_folder, "directions")
        # lbl_folder = os.path.join(data_folder, "labels")


        self.transform = transforms.Compose([transforms.ToTensor()])
        # self.positions, self.directions, self.labels = np.array([]),np.array([]) ,np.array([]) ,np.array([])  
        self.images = []
        self.positions, self.directions, self.labels = [], [], []
        self.crosses = []
        if data_type == 'train':
            if data_inds == None:
                num_inds = len(os.listdir(img_folder))
                data_inds = random_inds(num_inds, num_inds)

            if 'pos ' in net_type:
                for i in data_inds:
                    img = cv2.imread(os.path.join(img_folder, '%06d.png'%i))
                    self.images.append(img)
                    
                    self.images.append(os.path.join(img_folder, '%06d.png'%i))
                    self.positions.append(os.path.join(pos_folder, '%06d.npy'%i))

            elif 'dir' in net_type:
                for i in data_inds:
                    dir_labels = np.load(os.path.join(dir_folder, "%06d.npy"%i))
                    self.images.extend([os.path.join(img_folder, '%06d.png'%i) for _ in range(16)])                 
                    self.positions.extend([os.path.join(pos_folder, '%06d.npy'%i) for _ in range(16)])
                    self.directions.extend(self.dir_list)
                    self.labels.extend(dir_labels)
                    
                    self.crosses.extend([os.path.join(crs_folder, '%06d.png'%i) for _ in range(16)])                 
        elif data_type == 'test':
            for f in os.listdir(data_folder):
                if f.split('.')[-1] == 'png': self.images.append(os.path.join(data_folder, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        if self.data_type == 'train':
            img = cv2.resize(cv2.imread(self.images[index]), (self.img_w, self.img_h))
            crossing = cv2.resize(cv2.imread(self.crosses[index], 0), (self.img_w, self.img_h))
            p = np.load(self.positions[index]) # pull, hold
            p_no_hold = np.array([p[0]])
            
            heatmap = gauss_2d_batch(self.img_h, self.img_w, self.sigma, p)
            heatmap_no_hold = gauss_2d_batch(self.img_h, self.img_w, self.sigma, p_no_hold)
            
            img = self.transform(img)
            crossing = self.transform(crossing)

            if 'pos ' in self.net_type: return img, heatmap
            
            elif 'dir' in self.net_type:
                # cat_img = torch.cat((img, heatmap_no_hold))
                cat_img = torch.cat((img, heatmap_no_hold, crossing))
                q = torch.from_numpy(self.directions[index])
                l = torch.tensor(self.labels[index])
                return cat_img, q, l

        elif self.data_type == 'test':
            img = cv2.resize(cv2.imread(self.images[index]), (self.img_w, self.img_h))
            img = self.transform(img)
            return img
        
            # return all_img_no_hold, q, l


if __name__ == "__main__":

    # data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\HoldAndPullDirectionDataAll"
    data_folder = 'D:\\Dataset\\sepnet\\val'
    inds = random_inds(2, 100)

    # train_dataset = SepDataset(512, 512, data_folder, net_type='dir', data_inds=inds)
    train_dataset = SepDataset(512, 512, data_folder, net_type='dir')
    print(len(train_dataset))
    loader = DataLoader(train_dataset)
    print(len(loader))
    # for i in range(len(train_dataset)):
    #     img, direction, lbl = train_dataset[i]
    #     print(img.shape, direction.shape, lbl.shape)
    #     direction = direction.detach().cpu().numpy()

    #     print(np.round(vector2direction(direction),1), lbl)

    # for i in range(len(train_dataset)):
    #     data = train_dataset[i]
    #     for d in data:
    #         print(d.shape)

    

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