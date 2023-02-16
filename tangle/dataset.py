"""
Dataset class
    `PickDataset` for PickNet
    `PullDataset` for PullNet
Author: xinyi
Date: 20220517
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tangle.utils import *

class PickDataset(Dataset):
    """
    Output: 
        torch.Size([3, W, H]) - 3 channel image
        torch.Size([2, W, H]) - 2 channel mask: pick + tangle
    """
    def __init__(self, img_width, img_height, data_folder, data_inds=None, data_type='train'):
        self.img_w = img_width
        self.img_h = img_height
        self.data_type = data_type

        img_folder = os.path.join(data_folder, "images")
        msk_folder = os.path.join(data_folder, "masks")
        # gsp_folder = os.path.join(data_folder, "grasps")
        lbl_path = os.path.join(data_folder, "labels.npy")

        self.transform = transforms.Compose([transforms.ToTensor()])
        labels = np.load(lbl_path) 

        self.images = []
        self.masks = []
        self.labels = []
        # self.grasps = []
        if data_type == 'train':
            if data_inds == None:
                num_inds = len(os.listdir(img_folder))
                data_inds = random_inds(num_inds, num_inds)
            for loaded, i in enumerate(data_inds): 
                self.images.append(os.path.join(img_folder, "%06d.png"%i))
                self.masks.append(os.path.join(msk_folder, "%06d.png"%i))
                self.labels.append(labels[i])

                print('Loading data: %6d / %6d' % (loaded, len(data_inds)), end='')
                print('\r', end='')
            print(f"Finish loading {len(data_inds)} samples! ") 
        
        elif data_type == 'test':

            for f in os.listdir(data_folder):
                if f.split('.')[-1] == 'png': self.images.append(os.path.join(data_folder, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        if self.data_type == 'train':            
                        
            src_img = cv2.imread(self.images[index])
            src_msk = cv2.imread(self.masks[index], 0)
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
            src_img = cv2.resize(src_img, (self.img_w, self.img_h))
            return self.transform(src_img)

class PullDataset(Dataset):
    """
    Output: 
        torch.Size([3, W, H]) - 3 channel image 
        torch.Size([1, W, H]) - pull gauss
    """
    def __init__(self, img_width, img_height, data_folder, sigma=8, data_type="train", data_inds=None):
        self.img_w = img_width
        self.img_h = img_height
        self.sigma = sigma
        self.data_type = data_type
                
        images_folder = os.path.join(data_folder, "images")
        masks_folder = os.path.join(data_folder, "masks")
        positions_path = os.path.join(data_folder, "positions.npy")

        data_num = len(os.listdir(images_folder))
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.images, self.masks = [], []
        self.positions, self.directions = [], []
        positions = np.load(positions_path)
        
        if data_inds == None:
            data_inds = list(range(data_num))

        if data_type == "train":
            for loaded, i in enumerate(data_inds):
                self.images.append(os.path.join(images_folder, "%06d.png" % i))
                self.masks.append(os.path.join(masks_folder, "%06d.png" % i))
                self.positions.append(positions[i])

                print('Loading data: %d / %d' % (loaded, len(data_inds)), end='')
                print('\r', end='') 

            print(f"Finish loading {len(data_inds)} samples! ")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        src_img = cv2.imread(self.images[index])
        src_p = self.positions[index]
        src_h, src_w, _ = src_img.shape

        src_msk = cv2.imread(self.masks[index], 0)
        msk = cv2.resize(src_msk, (self.img_w, self.img_h))
        
        p = np.array([src_p[0]*self.img_w/src_w, src_p[1]*self.img_h/src_h])
        img = cv2.resize(src_img, (self.img_w, self.img_h))
        inp = self.transform(img)
        out = gauss_2d_batch(self.img_w, self.img_h, sigma=8, locs=[p])

        return inp, out

if __name__ == "__main__":

    # ---------------------- PickNet Dataset -------------------

    pn_data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\picknet_dataset"
    inds = random_inds(10, 80000) 
    train_dataset = PickDataset(512,512,pn_data_folder)
    for i in range(len(train_dataset)):
        img, msk = train_dataset[i]
        depth = visualize_tensor(img)
        pick_m = visualize_tensor(msk[0])
        sep_m = visualize_tensor(msk[1])
        # grasp = visualize_tensor(msk[2],cmap=True)

        if train_dataset.labels[i] == 1: 
            label = "Label: separate"
            sep_cnt, _ = cv2.findContours(sep_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            heatmap=cv2.drawContours(depth,sep_cnt,-1, [255,102,51] ,2) # blue
        else:
            label = "Label: pick"
            pick_cnt, _ = cv2.findContours(pick_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            heatmap=cv2.drawContours(depth,pick_cnt,-1, [144,89,244],2) # pink

        cv2.imshow(label, heatmap)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # ---------------------- PullNet Dataset -------------------

    sn_data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\pullnet_dataset"
    inds = random_inds(10, 20000) 
    train_dataset = PullDataset(512,512,sn_data_folder, data_inds=inds)
    for i in range(len(train_dataset)):
        inp, out = train_dataset[i]
        print(inp.shape, out.shape)
        inp = visualize_tensor(inp)
        out = visualize_tensor(out, cmap=True)
        overlay = cv2.addWeighted(inp, 0.5, out, 0.5, 1)
        cv2.imshow("w", overlay)
        cv2.waitKey()
        cv2.destroyAllWindows()

        if i > 10: break
