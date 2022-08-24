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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from tangle.utils import *

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# transform = transforms.Compose(
#     [
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class PickDataset(Dataset):
    """
    Output: 
        torch.Size([3, W, H]) - 3 channel image
        torch.Size([2, W, H]) - 2 channel mask: pick + tangle
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

# class SepDataset(Dataset):
#     """
#     Output: 
#         torch.Size([5, 512, 512]) - 3 channel image + pull gauss + hold gauss
#         torch.Size([2]) - normalized vector for direction, where [0,0] point to right
#         Scalar - label: 0/1
#     Usage: 
#         if 'sep_pos' - 3 channel image --> 2 channel mask
#         if 'sep_dir' - 3+2 channel image + direction --> label
#     """
#     def __init__(self, img_height, img_width, data_folder, net_type, sigma=6, data_type='train', data_inds=None):
#         self.img_h = img_height
#         self.img_w = img_width
#         self.net_type = net_type
#         self.sigma = sigma
#         self.data_type = data_type
#         self.dir_list = []
#         for i in range(16):
#             self.dir_list.append(angle2vector(i*360/16))

#         img_folder = os.path.join(data_folder, "images")
#         pos_folder = os.path.join(data_folder, "positions")
#         crs_folder = os.path.join(data_folder, "heatmaps")
#         dir_folder = os.path.join(data_folder, "directions")
#         # lbl_folder = os.path.join(data_folder, "labels")


#         self.transform = transforms.Compose([transforms.ToTensor()])
#         # self.positions, self.directions, self.labels = np.array([]),np.array([]) ,np.array([]) ,np.array([])  
#         self.images = []
#         self.positions, self.directions, self.labels = [], [], []
#         self.crosses = []
#         if data_type == 'train':
#             if data_inds == None:
#                 num_inds = len(os.listdir(img_folder))
#                 data_inds = random_inds(num_inds, num_inds)

#             if 'pos ' in net_type:
#                 # for i in data_inds:
#                 for i in tqdm.tqdm(data_inds, f"Processing dataset", ncols=100):
#                     img = cv2.imread(os.path.join(img_folder, '%06d.png'%i))
#                     self.images.append(img)
                    
#                     self.images.append(os.path.join(img_folder, '%06d.png'%i))
#                     self.positions.append(os.path.join(pos_folder, '%06d.npy'%i))

#             elif 'dir' in net_type:
#                 # for i in data_inds:
#                 for i in tqdm.tqdm(data_inds, f"Processing dataset", ncols=100):
#                     dir_labels = np.load(os.path.join(dir_folder, "%06d.npy"%i))
#                     self.images.extend([os.path.join(img_folder, '%06d.png'%i) for _ in range(16)])                 
#                     self.positions.extend([os.path.join(pos_folder, '%06d.npy'%i) for _ in range(16)])
#                     self.directions.extend(self.dir_list)
#                     self.labels.extend(dir_labels)
                    
#                     self.crosses.extend([os.path.join(crs_folder, '%06d.png'%i) for _ in range(16)])                 
#         elif data_type == 'test':
#             for f in os.listdir(data_folder):
#                 if f.split('.')[-1] == 'png': self.images.append(os.path.join(data_folder, f))

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):

#         if self.data_type == 'train':
#             img = cv2.resize(cv2.imread(self.images[index]), (self.img_w, self.img_h))
#             crossing = cv2.resize(cv2.imread(self.crosses[index], 0), (self.img_w, self.img_h))
#             p = np.load(self.positions[index]) # pull, hold
#             p_no_hold = np.array([p[0]])
            
#             heatmap = gauss_2d_batch(self.img_h, self.img_w, self.sigma, p)
#             heatmap_no_hold = gauss_2d_batch(self.img_h, self.img_w, self.sigma, p_no_hold)
            
#             img = self.transform(img)
#             crossing = self.transform(crossing)

#             if 'pos ' in self.net_type: return img, heatmap
            
#             elif 'dir' in self.net_type:
#                 # cat_img = torch.cat((img, heatmap_no_hold))
#                 cat_img = torch.cat((img, heatmap_no_hold, crossing))
#                 q = torch.from_numpy(self.directions[index])
#                 l = torch.tensor(self.labels[index])
#                 return cat_img, q, l

#         elif self.data_type == 'test':
#             img = cv2.resize(cv2.imread(self.images[index]), (self.img_w, self.img_h))
#             img = self.transform(img)
#             return img
        
#             # return all_img_no_hold, q, l

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
    def __init__(self, img_height, img_width, data_folder, net_type, sigma=6, data_type='train', data_inds=None):
        self.img_h = img_height
        self.img_w = img_width
        self.net_type = net_type
        self.sigma = sigma
        self.data_type = data_type
        self.dir_list = []
        
        images_folder = os.path.join(data_folder, "images")
        positions_path = os.path.join(data_folder, "positions.npy")
        labels_path = os.path.join(data_folder, "labels.npy")
        direction_path = os.path.join(data_folder, "direction.npy")

        data_num = len(os.listdir(images_folder))
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.images = []

        self.positions, self.directions, self.labels = [], [], []
        # self.crosses = []
        if data_type == 'train':
            if data_inds == None:
                # data_inds = random_inds(data_num, data_num)
                data_inds = list(range(data_num))

            if 'pos' in net_type:
                positions = np.load(positions_path)
                loaded_num = 0
                for i in data_inds:
                    self.images.append(os.path.join(images_folder, '%06d.png'%i))
                    self.positions.append(positions[i])
                    loaded_num += 1
                    print('Loading data: %d / %d' % (loaded_num, len(data_inds)), end='')
                    print('\r', end='') 
                print(f"Finish loading {loaded_num} samples! ")

            elif 'dir' in net_type:
                positions = np.load(positions_path)
                labels = np.load(labels_path)
                
                direction = np.load(direction_path)
                loaded_num = 0
                for i in data_inds:
                    for j in range(len(direction)):
                        self.images.append(os.path.join(images_folder, '%06d.png'%i))
                        self.positions.append(positions[i])
                        self.labels.append(labels[i][j])

                        self.directions.append(direction[j])
                        loaded_num += 1
                        print('Loading data: %d / %d' % (loaded_num, len(data_inds*len(direction))), end='')
                        print('\r', end='') 
                print(f"Finish loading {loaded_num} samples! ")

        elif data_type == 'test':
            for f in os.listdir(data_folder):
                if f.split('.')[-1] == 'png': self.images.append(os.path.join(data_folder, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.data_type == 'train':
            _img = cv2.imread(self.images[index]) 
            _p = self.positions[index]
            # cv2.circle(_img, _p[0], 7, (0,255,0), 2)
            # cv2.imshow("", _img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            _img = cv2.imread(self.images[index])
            _p = self.positions[index] # pull, hold
            _h, _w, _ = _img.shape

            img = cv2.resize(_img, (self.img_w, self.img_h))
            p = _p.copy()
            p[:,0] = _p[:,0] * self.img_w / _w
            p[:,1] = _p[:,1] * self.img_h / _h
            p = p.astype(int)
            self.positions[index] = p
            heatmap = gauss_2d_batch(self.img_h, self.img_w, self.sigma, p)
            
            # p_no_hold = np.array([p[0]])
            # heatmap_no_hold = gauss_2d_batch(self.img_h, self.img_w, self.sigma, p_no_hold)
            
            img = self.transform(img)

            if 'pos' in self.net_type: 
                return img, heatmap
            
            elif 'dir' in self.net_type:
                # cat_img = torch.cat((img, heatmap_no_hold))
                cat_img = torch.cat((img, heatmap))
                q = torch.from_numpy(self.directions[index])
                l = torch.tensor(self.labels[index])
                return cat_img, q, l

        elif self.data_type == 'test':
            img = cv2.resize(cv2.imread(self.images[index]), (self.img_w, self.img_h))
            img = self.transform(img)
            return img
        
            # return all_img_no_hold, q, l
class SepDatasetAM(Dataset):
    def __init__(self, img_h, img_w, data_folder, sigma=8, data_type="train", data_inds=None):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.sigma = sigma
        self.data_type = data_type
        degrees = []
        
        
        images_folder = os.path.join(data_folder, "images")
        positions_path = os.path.join(data_folder, "positions.npy")
        labels_path = os.path.join(data_folder, "labels.npy")
        direction_path = os.path.join(data_folder, "direction.npy")

        data_num = len(os.listdir(images_folder))
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.images = []
        self.pullmaps = []
        self.holdmaps = []
        from bpbot.utils import rotate_img
        self.positions, self.directions = [], []
        if data_type == "train":
            if data_inds == None:
                data_inds = list(range(data_num))
            positions = np.load(positions_path)
            labels = np.load(labels_path)
            direction = np.load(direction_path)
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

                [pullmap, holdmap] = gauss_2d_batch(self.img_h, self.img_w, self.sigma, p)
                pullmap = visualize_tensor(pullmap)
                holdmap = visualize_tensor(holdmap)
                for j in np.where(labels[i]==1)[0]:
                    theta = degrees[j]
                    self.images.append(rotate_img(img, theta))
                    self.pullmaps.append(rotate_img(pullmap, theta))
                    self.holdmaps.append(rotate_img(holdmap, theta))
                    
                    loaded_num += 1
                    print('Loading data: %d / %d' % (loaded_num, len(data_inds)), end='')
                    print('\r', end='') 
            print(f"Finish loading {loaded_num} samples! ")
    def __len__(self): 
        return len(self.images)

    def __getitem__(self, index):
        if self.data_type == "train": 
            inp = torch.cat((self.transform(self.images[index]), self.transform(self.holdmaps[index])))
            out = self.transform(self.pullmaps[index]).double()
            return inp, out

class SepMultiDataset(Dataset):
    def __init__(self, img_h, img_w, folder, sigma=6, data_inds=None):
        self.img_h = img_h
        self.img_w = img_w
        self.sigma = sigma
        images_folder = os.path.join(folder, "images")
        positions_path = os.path.join(folder, "positions.npy")
        labels_path = os.path.join(folder, "labels.npy")
        if data_inds is None: 
            data_num = len(os.listdir(images_folder))
            data_inds = random_inds(data_num, data_num)
        
        self.transform = transforms.Compose([transforms.ToTensor()])
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

if __name__ == "__main__":

    from tangle import Config
    cfg = Config(config_type="train")
    # data_folder = cfg.data_dir
# 

    # data_inds = random_inds(10,1000)
    data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorAugment"
    data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEight"
    # ---------------------- SepNet-D Dataset -------------------
    # train_dataset = SepDataset(512,512,data_folder,net_type="sep_pos")
    # print(len(train_dataset))
    # inds = random_inds(10, len(train_dataset))
    # for i in inds:
    #     img, hmap = train_dataset[i]
    #     depth = visualize_tensor(img)
    #     pull = visualize_tensor(hmap[0], cmap=True)
    #     hold = visualize_tensor(hmap[1], cmap=True)
    #     pull_map = cv2.addWeighted(depth, 0.65, pull, 0.35, 1)
    #     hold_map = cv2.addWeighted(depth, 0.65, hold, 0.35, 1)
    #     cv2.imshow("", cv2.hconcat([depth, pull_map, hold_map]))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    # ---------------------- SepNet-D Dataset -------------------
    # train_dataset = SepDataset(500,500,data_folder,net_type="sep_dir", sigma=9)
    # for i in inds:
    #     img, v, l = train_dataset[i]
    #     p = train_dataset.positions[i]
    #     print(train_dataset.images[i], p.flatten())
    #     # print(img.shape, p.flatten(), v, l)
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

    inds = random_inds(20, 1000) 
    train_dataset = SepDatasetAM(500,500,data_folder,sigma=9, data_inds=inds)
    print(len(train_dataset.images))
    print(len(train_dataset.directions))
    for i in range(20):
        print("=>", i)
        out1, out2 = train_dataset[i]

        # print(img.shape, pull.shape, hold.shape)
        img = visualize_tensor(out1[:3])
        pull = visualize_tensor(out1[-1], cmap=True)
        hold = visualize_tensor(out2, cmap=True)
        print(img.shape, pull.shape, hold.shape)
        pull_map = cv2.addWeighted(img, 0.65, pull, 0.35, 1)
        hold_map = cv2.addWeighted(img, 0.65, hold, 0.35, 1)
        cv2.imshow("", cv2.hconcat([img, pull_map, hold_map]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imshow("", cv2.hconcat((pull, hold)))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # cv2.circle(_img, _p[0], 7, (0,255,0), 2)

    # for tr, p in zip(train_dataset, train_dataset.positions):
    #     img, v, l = tr

    # ---------------------- SepMultiDataset -------------------
    # train_dataset = SepMultiDataset(512, 512, data_folder, sigma=8)
    # print(len(train_dataset))
    # inds = random_inds(10, len(train_dataset))
    # for i in inds:
    #     img, lbl = train_dataset[i]
    #     print(img.shape, lbl.shape)
    #     depth = visualize_tensor(img[0:3])
    #     pull = visualize_tensor(img[3], cmap=True)
    #     hold = visualize_tensor(img[4], cmap=True)
    #     pull_map = cv2.addWeighted(depth, 0.65, pull, 0.35, 1)
    #     hold_map = cv2.addWeighted(depth, 0.65, hold, 0.35, 1)
    #     drawn = draw_vectors_bundle(depth, (250,250), scores=lbl)
    

    # # data_folder = "D:\\datasets\\holdandpull_test"

    # test_dataset = PickDataset(512, 512, data_folder, data_type='test')
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    # print(test_dataset.images)
    # for s in test_loader:
    #     print(s.shape)
    # img, direction, lbl = train_dataset[23]
    # for i in range(16):
    #     img, direction, lbl = train_dataset[i]
    #     v = direction.detach().cpu().numpy()
    #     s = lbl.detach().cpu().numpy()
    #     print(v,s)
    #     depth = visualize_tensor(img[0:3,])
    #     if s == 1: draw_vector(depth, (250,250), v, color=(0,255,0))
    #     else: draw_vector(depth, (250,250), v, color=(255,0,0))
    #     pull_map = visualize_tensor(img[3,:], cmap=True)
    #     hold_map = visualize_tensor(img[4,:], cmap=True)
    #     pull_vis = cv2.addWeighted(depth, 0.65, pull_map, 0.35, 1)
    #     hold_vis = cv2.addWeighted(depth, 0.65, hold_map, 0.35, 1)

    #     cv2.imshow("", cv2.hconcat([pull_vis, hold_vis]))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # s_num = 0
    # label_f = os.path.join(data_folder, "labels") 
    # for d in tqdm.tqdm(os.listdir(label_f), "Proccessing", ncols=100):
    #     if np.load(os.path.join(label_f, d))[0]: 
    #         s_num += 1
    # print(s_num, "/", len(os.listdir(label_f)) )
    # inds = random_inds(20,100000)
    # train_data = PickDataset(512,512,data_folder, data_type="train", data_inds=inds)
    # for i in range(len(train_data)):
    #     img, msks = train_data[i]
    #     img = visualize_tensor(img)
    #     msk_pick = visualize_tensor(msks[0], cmap=True)
    #     msk_sep = visualize_tensor(msks[1], cmap=True)
    #     vis1 = cv2.addWeighted(img, 0.65, msk_pick, 0.35, 1)
    #     vis2 = cv2.addWeighted(img, 0.65, msk_sep, 0.35, 1)
    #     cv2.imshow("", cv2.hconcat([vis1, vis2]))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
        # if i > 20: break



    # data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\HoldAndPullDirectionDataAll"
    # data_folder = 'D:\\Dataset\\sepnet\\val'
    # inds = random_inds(2, 100)

    # # train_dataset = SepDataset(512, 512, data_folder, net_type='dir', data_inds=inds)
    # train_dataset = SepDataset(512, 512, data_folder, net_type='dir')
    # print(len(train_dataset))
    # loader = DataLoader(train_dataset)
    # print(len(loader))
    # for i in range(len(train_dataset)):
    #     img, direction, lbl = train_dataset[i]
    #     print(img.shape, direction.shape, lbl.shape)
    #     direction = direction.detach().cpu().numpy()

    #     print(np.round(vector2direction(direction),1), lbl)

