"""
Dataset class
    `PickDataset` for PickNet
    `SepDataset` for SepPositionNet, SepDirectionNet
Author: xinyi
Date: 20220517
"""
import os
import json
import tqdm
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
        gsp_folder = os.path.join(data_folder, "grasps")
        lbl_path = os.path.join(data_folder, "labels.npy")

        self.transform = transforms.Compose([transforms.ToTensor()])
        labels = np.load(lbl_path) 

        self.images = []
        self.masks = []
        self.labels = []
        self.grasps = []
        if data_type == 'train':
            if data_inds == None:
                num_inds = len(os.listdir(img_folder))
                data_inds = random_inds(num_inds, num_inds)
            for i in data_inds: 
                self.images.append(os.path.join(img_folder, "%06d.png"%i))
                self.masks.append(os.path.join(msk_folder, "%06d.png"%i))
                self.grasps.append(os.path.join(gsp_folder, "%06d.png" % i))
                self.labels.append(labels[i][0])

                # self.points.append(np.asarray([labels[i][1:3],labels[i][4:6]]).astype(int))
                # self.labels.append(np.load(os.path.join(lbl_folder, "%06d.npy"%i))[0])
                print('Loading data: %6d / %6d' % (i, len(data_inds)), end='')
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
            src_gsp = cv2.imread(self.grasps[index], 0)
            # # check the size
            # src_h, src_w = src_msk.shape
            src_img = cv2.resize(src_img, (self.img_w, self.img_h))
            src_msk = cv2.resize(src_msk, (self.img_w, self.img_h))
            src_gsp = cv2.resize(src_gsp, (self.img_w, self.img_h))

            black = np.zeros((self.img_h,self.img_w), dtype=np.uint8)
            
            if self.labels[index] == 0:
                img = self.transform(src_img)
                msk = self.transform(np.stack([src_msk, black, src_gsp], 2))
                
            else:
                img = self.transform(src_img)
                msk = self.transform(np.stack([black, src_msk, src_gsp], 2))
            return img, msk

        elif self.data_type == 'test':
            src_img = cv2.imread(self.images[index])
            # src_h, src_w,_ = src_img.shape
            # if src_h != self.img_h or src_w != self.img_h:
            src_img = cv2.resize(src_img, (self.img_w, self.img_h))
            return self.transform(src_img)

class SepDataset(Dataset):
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
        # option 1: mask
        # out = self.transform(msk)
        # optiopn 2: gaussian 2d for points
        out = gauss_2d_batch(self.img_w, self.img_h, sigma=8, locs=[p])

        return inp, out

if __name__ == "__main__":

    from tangle import Config
    cfg = Config(config_type="train")

    data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorAugment"
    data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEight"
    BLUE = [51,102,255]
    PINK = [244,89,144]
    BLUE_RV = [255,102,51]
    PINK_RV = [144,89,244]

    # ---------------------- PickNet Dataset -------------------
    # data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\PickDataNew"
    # Np, Nn = 0, 0
    # train_dataset = PickDataset(512,512,data_folder)
    # for i in range(len(train_dataset)):
    #     if train_dataset.labels[i] == 1: 
    #         label = "Label: separate"
    #     else:
    #         label = "Label: pick"
    #     img, msk = train_dataset[i]
    #     depth = visualize_tensor(img)
    #     pick_m = visualize_tensor(msk[0])
    #     sep_m = visualize_tensor(msk[1])
    #     grasp = visualize_tensor(msk[2],cmap=True)
    #     pick = cv2.cvtColor(pick_m, cv2.COLOR_GRAY2RGB)
    #     sep = cv2.cvtColor(sep_m, cv2.COLOR_GRAY2RGB)
    #     pick[pick_m > 0] = PINK_RV
    #     sep[sep_m > 0] = BLUE_RV

    #     pick_map = cv2.addWeighted(depth, 0.5, pick, 0.5, 1)
    #     sep_map = cv2.addWeighted(depth, 0.5, sep, 0.5, 1)
    #     grasp_map = cv2.addWeighted(depth, 0.5, grasp, 0.5, 1)
    #     cv2.imshow(label, cv2.hconcat([depth, pick_map, sep_map, grasp_map]))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    #     if i > 10: break

    # ---------------------- SepNet-P Dataset -------------------
    # inds = random_inds(10, 1000) 
    data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataNew"
    train_dataset = SepDataset(512,512,data_folder)
    # i=0
    # print(train_dataset[1])
    for data in train_dataset:
        inp, out = data
        print(inp.shape, out.shape)
        inp = visualize_tensor(inp)
        out = visualize_tensor(out, cmap=True)
        plt.imshow(inp)
        plt.imshow(out, alpha=0.3)
        plt.show()
    #     i+=1
    #     if i > 10: break
