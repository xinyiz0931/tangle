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
            for i in tqdm.tqdm(data_inds, f"Processing dataset", ncols=100):
            # for i in data_inds: 
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
        torch.Size([5, W, H]) - 3 channel image + pull gauss + hold gauss
        torch.Size([2]) - normalized vector for direction, where [0,0] point to right
        Scalar - label: 0/1
    Usage: 
        if 'sep_pos' - 3 channel image --> 2 channel mask
        if 'sep_dir' - 3+2 channel image + direction --> label
    """
    def __init__(self, img_width, img_height, data_folder, net_type, sigma=8, data_type="train", data_inds=None):
        self.img_w = img_width
        self.img_h = img_height
        self.net_type = net_type
        self.sigma = sigma
        self.data_type = data_type
        
        
        images_folder = os.path.join(data_folder, "images")
        positions_path = os.path.join(data_folder, "positions.npy")
        labels_path = os.path.join(data_folder, "labels.npy")
        direction_path = os.path.join(data_folder, "direction.npy")

        data_num = len(os.listdir(images_folder))
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.images = []
        self.positions, self.directions, self.labels = [], [], []
        self.pullmaps, self.holdmaps = [], []

        positions = np.load(positions_path)
        labels = np.load(labels_path)
        direction = np.load(direction_path)
        
        if data_inds == None:
            data_inds = list(range(data_num))

        if net_type == "sep_pos" or net_type == "sep_hold":
             
            loaded_num = 0
            for i in data_inds:
                _img = cv2.imread(os.path.join(images_folder, "%06d.png"%i))
                _p = positions[i] # pull, hold
                _h, _w, _ = _img.shape
                img = cv2.resize(_img, (self.img_w, self.img_h))
                p = _p.copy()
                p[:,0] = _p[:,0] * self.img_w / _w
                p[:,1] = _p[:,1] * self.img_h / _h
                p = p.astype(int)

                [pullmap, holdmap] = gauss_2d_batch(self.img_w, self.img_h, self.sigma, p, use_torch=False)
                self.images.append(img)
                self.pullmaps.append(pullmap)
                self.holdmaps.append(holdmap)
                
                loaded_num += 1
                print('Loading data: %d / %d' % (loaded_num, len(data_inds)), end='')
                print('\r', end='') 
            print(f"Finish loading {loaded_num} samples! ")

        elif net_type == "sep_dir":
            degrees = []
            for i in range(len(direction)):
                degrees.append(360/len(direction)*i)
            loaded_num = 0
            for i in data_inds:
                _img = cv2.imread(os.path.join(images_folder, "%06d.png"%i))
                _p = positions[i] # pull, hold
                _h, _w, _ = _img.shape
                img = cv2.resize(_img, (self.img_w, self.img_h))
                p = _p.copy()
                p[:,0] = _p[:,0] * self.img_w / _w
                p[:,1] = _p[:,1] * self.img_h / _h
                p = p.astype(int)

                [pullmap, holdmap] = gauss_2d_batch(self.img_w, self.img_h, self.sigma, p, use_torch=False)
                for j in range(len(direction)):
                    theta = degrees[j]
                    self.images.append(img)
                    self.pullmaps.append(pullmap)
                    self.holdmaps.append(holdmap)
                    self.directions.append(direction[j])
                    self.labels.append(labels[i][j])
                    self.positions.append(_p)
                    loaded_num += 1
                    print('Loading data: %d / %d' % (loaded_num, len(data_inds)), end='')
                    print('\r', end='') 

            print(f"Finish loading {loaded_num} samples! ")

        elif net_type == "sep_pull": 
            if data_type == "train":

                from bpbot.utils import rotate_img
                
                degrees = []
                for i in range(len(direction)):
                    degrees.append(360/len(direction)*i)
                loaded_num = 0
                for i in data_inds:
                    _img = cv2.imread(os.path.join(images_folder, "%06d.png"%i))
                    _p = positions[i] # pull, hold
                    _h, _w, _ = _img.shape
                    img = cv2.resize(_img, (self.img_w, self.img_h))
                    p = _p.copy()
                    p[:,0] = _p[:,0] * self.img_w / _w
                    p[:,1] = _p[:,1] * self.img_h / _h
                    p = p.astype(int)

                    [pullmap, holdmap] = gauss_2d_batch(self.img_w, self.img_h, self.sigma, p, use_torch=False)
                    for j in np.where(labels[i]==1)[0]:
                        theta = degrees[j]
                        self.images.append(rotate_img(img, theta))
                        self.pullmaps.append(rotate_img(pullmap, theta))
                        self.holdmaps.append(rotate_img(holdmap, theta))
                        
                        loaded_num += 1
                        print('Loading data: %d / %d' % (loaded_num, len(data_inds)), end='')
                        print('\r', end='') 

                print(f"Finish loading {loaded_num} samples! ")
        
            elif data_type == 'val':
                for i in data_inds:
                    _img = cv2.imread(os.path.join(images_folder, "%06d.png" % i))
                    _p = positions[i]
                    _h, _w, _ = _img.shape
                    img = cv2.resize(_img, (self.img_w, self.img_h))
                    p = _p.copy()
                    p[:,0] = _p[:,0] * self.img_w / _w
                    p[:,1] = _p[:,1] * self.img_h / _h
                    p = p.astype(int)
                    
                    [pullmap, holdmap] = gauss_2d_batch(self.img_w, self.img_h, self.sigma, p, use_torch=False)
                    self.images.append(img)
                    self.holdmaps.append(holdmap)
                    self.pullmaps.append(pullmap)
                    self.labels.append(labels[i]) 

                    print('Loading data: %d / %d' % (i, len(data_inds)), end='')
                    print('\r', end='') 
            print(f"Finish loading {len(data_inds)} samples! ")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.net_type == "sep_pos" or self.net_type == "sep_hold":
            inp = self.transform(self.images[index])
            out = torch.cat((self.transform(self.pullmaps[index]), self.transform(self.holdmaps[index])))
            return inp, out

        elif self.net_type == "sep_dir":
            inp1 = torch.cat((self.transform(self.images[index]), self.transform(self.pullmaps[index]), self.transform(self.holdmaps[index])))
            inp2 = torch.from_numpy(self.directions[index])
            out = torch.tensor(self.labels[index])
            return inp1, inp2, out

        elif self.net_type == "sep_pull":
            if self.data_type == "train":
                inp = torch.cat((self.transform(self.images[index]), self.transform(self.holdmaps[index])))
                out = self.transform(self.pullmaps[index]).double()
                return inp, out
            elif self.data_type == "val":
                inp = torch.cat((self.transform(self.images[index]), self.transform(self.holdmaps[index])))
                out = self.transform(self.pullmaps[index])
                return inp, out, self.labels[index] 

if __name__ == "__main__":

    from tangle import Config
    cfg = Config(config_type="train")

    data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorAugment"
    data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEight"
    data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorReal"

    # ---------------------- SepNet-P Dataset -------------------
    inds = random_inds(10, 1000) 
    train_dataset = SepDataset(512,512,data_folder,net_type="sep_pos", data_inds=inds)
    for i in range(10):
        img, hmap = train_dataset[i]
        depth = visualize_tensor(img)
        pull = visualize_tensor(hmap[0], cmap=True)
        hold = visualize_tensor(hmap[1], cmap=True)
        pull_map = cv2.addWeighted(depth, 0.65, pull, 0.35, 1)
        hold_map = cv2.addWeighted(depth, 0.65, hold, 0.35, 1)
        cv2.imshow("", cv2.hconcat([depth, hold_map, pull_map]))
        cv2.waitKey()
        cv2.destroyAllWindows()

    # ---------------------- SepNet-D Dataset -------------------
    # inds = random_inds(10, 1000) 
    # train_dataset = SepDataset(500,500,data_folder,net_type="sep_dir", data_type="train", data_inds=inds)
    # for i in range(10):
    #     img, v, l = train_dataset[i]
    #     p = train_dataset.positions[i]
    #     depth = visualize_tensor(img[0:3])
    #     pullmap = cv2.addWeighted(depth, 0.65, visualize_tensor(img[-2], cmap=True), 0.35, -1)
    #     if l == 0:
    #         pullmap = draw_vector(pullmap, p[0], visualize_tensor(v), color=(255,0,0))
    #     else:
    #         pullmap = draw_vector(pullmap, p[0], visualize_tensor(v), color=(0,255,0))
    #     holdmap = cv2.addWeighted(depth, 0.65, visualize_tensor(img[-1], cmap=True), 0.35, -1)
    #     cv2.imshow("", cv2.hconcat([depth, pullmap, holdmap]))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    # ---------------- SepNet Dataset AM + data_type="train" -----------------
    # inds = random_inds(2, 1000) 
    # train_dataset = SepDataset(500,500,data_folder, net_type="sep_pull", data_type="train", data_inds=inds)
    # for i in range(2):
    #     out1, out2 = train_dataset[i]

    #     print(i, "=>", out1.shape, out2.shape)
    #     img = visualize_tensor(out1[:3])
    #     pull = visualize_tensor(out1[-1], cmap=True)
    #     hold = visualize_tensor(out2, cmap=True)
    #     print(img.shape, pull.shape, hold.shape)
    #     pull_map = cv2.addWeighted(img, 0.65, pull, 0.35, 1)
    #     hold_map = cv2.addWeighted(img, 0.65, hold, 0.35, 1)
    #     cv2.imshow("", cv2.hconcat([img, pull_map, hold_map]))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # ---------------- SepNet Dataset AM + data_type="val" -----------------
    # inds = random_inds(2, 1000) 
    # train_dataset = SepDataset(500,500,data_folder, net_type="sep_pull", sigma=9, data_type="val", data_inds=inds)
    # for i in range(2):
    #     out1, out2, out3 = train_dataset[i]

    #     print(i, "=>", out1.shape, out2.shape, out3)
    #     img = visualize_tensor(out1[:3])
    #     hold = visualize_tensor(out1[3], cmap=True)
    #     pull = visualize_tensor(out2, cmap=True)
    #     pull_map = cv2.addWeighted(img, 0.65, pull, 0.35, 1)
    #     hold_map = cv2.addWeighted(img, 0.65, hold, 0.35, 1)
    #     cv2.imshow("", cv2.hconcat([img, hold_map, pull_map]))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()



