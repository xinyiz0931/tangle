import os
from random import sample
import sys
import logging
import timeit
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tqdm
from tangle.utils import *
from tangle import PickNet, SepPositionNet, SepDirectionNet
from tangle import PickDataset, SepDataset
from tangle.utils import *

class InferConfig(object):
    # choice = ['test', 'val']
    mode = 'test'
    # choice = ['pick', 'sep', 'sep_pos', 'sep_dir', 'pick_sep']
    infer_type = 'sep_pos'
    
    use_cuda = False
    batch_size = 1
    input_size = (512,512)

    root_dir = "/home/xpredictioninyi/Documents/"

    root_dir = "C:\\Users\\xinyi\\Documents"

    pick_ckpt = os.path.join(root_dir, 'Checkpoints', 'try8', 'model_epoch_9.pth')
    sep_pos_ckpt = os.path.join(root_dir, 'Checkpoints', 'try_38', 'model_epoch_2.pth')
    sep_dir_ckpt = os.path.join(root_dir, 'Checkpoints', 'try_SR', 'model_epoch_99.pth')

    if mode == 'test':
        if 'pick' in infer_type:
            dataset_dir = 'D:\\datasets\\picknet_dataset\\test'
        elif 'sep' in infer_type:
            dataset_dir = 'D:\\datasets\\sepnet_dataset\\test'
        else: 
            print(f"Wrong inference type: {infer_type} ... ")

    elif mode == 'val':
        if infer_type == 'pick':
            dataset_dir = 'D:\\datasets\\picknet_dataset\\val'

        elif infer_type == 'sep_pos' or infer_type == 'sep_dir':
            dataset_dir = 'D:\\datasets\\sepnet_dataset\\val'
        else:
            print(f"Wrong mode/inference combination: {mode}/{infer_type} ... ")
    else:
        print(f"Wrong mode: {mode} ... ")
    
    def __init__(self):
        pass
    
    def display(self):
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class Inference(object):

    def __init__(self, config):
        
        self.config = config
        self.use_cuda = config.use_cuda
        self.mode = config.mode
        self.infer_type = config.infer_type
        config.display()

        (img_h, img_w) = config.input_size
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = config.batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])

        # some combinations are not useful
        # if self.mode == 'val' and (self.infer_type == 'sep' or self.infer_type == 'pick_sep'):
        #     print('No such inference type using this mode ... ')
        #     return

        # check modelds' existence [pick, sep_pos, sep_dir]
        models = [False,False,False]

        if 'pick' in self.infer_type:
            self.picknet = PickNet(model_type='unet', out_channels=2)
            models[0] = True

        if 'sep' in self.infer_type:
            if self.infer_type == 'sep_pos':
                self.sepposnet = SepPositionNet(out_channels=2)
                models[1] = True

            elif self.infer_type == 'sep_dir':
                self.sepdirnet = SepDirectionNet(in_channels=5)
                models[2] = True
            else:
                self.sepposnet = SepPositionNet(out_channels=2)
                self.sepdirnet = SepDirectionNet(in_channels=5)
                models[1] = True
                models[2] = True
        
        if models[0] is True: 
            if self.use_cuda:
                self.picknet = self.picknet.cuda()
                self.picknet.load_state_dict(torch.load(self.config.pick_ckpt))
            else:
                self.picknet.load_state_dict(torch.load(self.config.pick_ckpt,map_location=torch.device("cpu")))  
        if models[1] is True:
            if self.use_cuda: 
                self.sepposnet = self.sepposnet.cuda()
                self.sepposnet.load_state_dict(torch.load(self.config.sep_pos_ckpt))
            else:
                self.sepposnet.load_state_dict(torch.load(self.config.sep_pos_ckpt,map_location=torch.device("cpu")))
        if models[2] is True:
            if self.use_cuda: 
                self.sepdirnet = self.sepdirnet.cuda()
                self.sepdirnet.load_state_dict(torch.load(self.config.sep_dir_ckpt))
            else:
                self.sepdirnet.load_state_dict(torch.load(self.config.sep_dir_ckpt,map_location=torch.device("cpu")))
            
        # if validation mode, it's necessary to load the dataset
        if self.mode == 'val':
            inds = random_inds(10,len(os.listdir(os.path.join(config.dataset_dir, "images"))))
            if 'pick' in self.infer_type:
                self.val_dataset = PickDataset(img_h, img_w, config.dataset_dir, data_inds=inds) 
            elif 'sep' in self.infer_type:
                self.val_dataset = SepDataset(img_h, img_w, config.dataset_dir, self.infer_type, data_inds=inds)
            self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size, shuffle=False)
        
        elif self.mode == 'test':
            self.dataset_dir = config.dataset_dir
            if 'pick' in self.infer_type:
                self.test_dataset = PickDataset(img_h, img_w, config.dataset_dir, data_type='test')
            elif 'sep' in self.infer_type:
                self.test_dataset = SepDataset(img_h, img_w, config.dataset_dir, self.infer_type, data_type='test')
            self.test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False)

    def get_grasps_for_sepnet(self, img):
        drawn = img.copy()
        grasps = [] # 2x2, [[pull_x, pull_y], [hold-x, hold_y]]
        def on_click(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(drawn,(x,y),5,(0,255,0),-1)
                print(f"{x},{y}")
                grasps.append([x,y])

        cv2.namedWindow('click twice to select pull and hold')
        cv2.setMouseCallback('click twice to select pull and hold',on_click)
        while(len(grasps)<2):
            cv2.imshow('click twice to select pull and hold',drawn)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or k==ord('q'):
                break
        cv2.destroyAllWindows()
        return grasps

    def get_image_list(self, data_path):
        """
        if data_path == *.png: detect for one image
        else data_path == directory: detect images in it
        return: a list contains all image paths 
        """
        data_list = []
        if os.path.isdir(data_path) and os.path.exists(data_path):  
            for f in os.listdir(data_path):
                if f.split('.')[-1] == 'png': data_list.append(os.path.join(data_path, f))
        elif os.path.isfile(data_path) and os.path.exists(data_path):
            if data_path.split('.')[-1] == 'png': 
                data_list.append(data_path)
        else:  
            print("Data path does not exist! ")
        return data_list

    def infer_pick(self, data_path=None):
        """
        infer picknet: classify two maps: pick/sep maps
        return: action_type: 'pick' or 'sep'
                location: [x,y]
        """
        grasps = [] # pick grasp point, sep grasp point
        action = 0 
        if data_path != None and not os.path.exists(data_path):
            print("Invalid path! ")
            return
        elif data_path != None and data_path.split('.')[-1] == 'png':
            # infer a single image
            img = cv2.resize(cv2.imread(data_path), (self.img_w, self.img_h))
            img_t = self.transform(img)
            img_t = torch.unsqueeze(img_t, 0).cuda() if self.use_cuda else torch.unsqueeze(img_t, 0)
            heatmap = self.picknet(img_t)[0]
            heatmap = heatmap.detach().cpu().numpy()

            for h in heatmap: 
                pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                grasps.append([pred_x, pred_y])
            
            if heatmap[0].max() < heatmap[1].max(): action = 1
            self.plot(data_path, heatmap, grasps)
            
        elif self.mode == 'test':
            if data_path != None: data_list = self.get_image_list(data_path)
            else: data_list = self.test_dataset.images

            start = timeit.default_timer()
            res_list = []
            for sample_batched in self.test_loader:
                sample_batched = sample_batched.cuda() if self.use_cuda else sample_batched
                heatmaps = self.picknet(sample_batched)
                for j in range(heatmaps.shape[0]):
                    
                    heatmap = heatmaps[j].detach().cpu().numpy()
                    res_list.append(heatmap)
            grasps = [] # pick grasp point, sep grasp point
            for d, res in zip(data_list, res_list):
                self.plot(d, res)

        elif self.mode == 'val':
            if data_path != None: data_list = self.get_image_list(data_path)
            else: data_list = self.val_dataset.images
               
            start = timeit.default_timer()
            res_list = []
            for sample_batched in self.val_loader:
                sample_batched = [Variable(d.cuda() if self.use_cuda else d) for d in sample_batched]
                img, mask_gt = sample_batched
                heatmaps = self.picknet(img)

                # get ground truth label
                if mask_gt[0].max() >=  mask_gt[1].max(): lbl_gt = 0
                else: lbl_gt = 1

                for j in range(heatmaps.shape[0]):
                    if mask_gt[j][0].max() >=  mask_gt[j][1].max(): lbl_gt = 0
                    else: lbl_gt = 1

                    h = heatmaps[j].detach().cpu().numpy()
                    if h[0].max() >= h[1].max(): lbl_pred = 0
                    else: lbl_pred = 1
                    res_list.append(h)

            for d, res in zip(data_list, res_list):
                self.plot(d, res)
            
            return

    # def plot_picknet(self, img_path, heatmaps):
    #     """
    #     plot picknet results for one image
    #     input: img_path [str], heatmaps: [2,H,W]
    #     """
    #     img = cv2.resize(cv2.imread(img_path), (self.img_w, self.img_h))

    #     splited = list(os.path.split(img_path))
    #     splited[-1] = 'out_' + splited[-1]
    #     splited.insert(-1, 'pred')

    #     if not os.path.exists(os.path.join(*splited[:-1])): 
    #         os.mkdir(os.path.join(*splited[:-1]))
    #     save_path = os.path.join(*splited)
        
    #     scores, overlays = [], []
    #     for h in heatmaps: 
    #         pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
    #         scores.append(h.max())
    #         vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #         vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    #         overlay = cv2.addWeighted(img, 0.7, vis, 0.3, 0)
    #         overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
    #         overlays.append(overlay)
        
    #     if scores[0] > scores[1]:
    #         cv2.putText(overlays[0], 'pick', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (48, 192, 19), 2)
    #         cv2.putText(overlays[1], 'sep', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    #     else:
    #         cv2.putText(overlays[0], 'pick', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    #         cv2.putText(overlays[1], 'sep', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (48, 192, 19), 2)
        
    #     # cv2.imwrite(save_path, cv2.hconcat(overlays))
    #     cv2.imshow('picknet result', cv2.hconcat(overlays))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    def infer_sep_pos(self, data_path=None):
        """
        infer sepnet-p: output two keypoints: hold, pull
        """
        if data_path != None and not os.path.exists(data_path):
            print("Invalid path!")
            return
        elif data_path != None and data_path.split('.')[-1] == 'png':
            # infer a single image
            print('use cuda? ', self.use_cuda)
            img = cv2.resize(cv2.imread(data_path), (self.img_w, self.img_h))
            img_t = self.transform(img)
            img_t = torch.unsqueeze(img_t, 0).cuda() if self.use_cuda else torch.unsqueeze(img_t, 0)
            heatmap = self.sepposnet.forward(img_t)[0] # (2,H,W)
            heatmap = heatmap.detach().cpu().numpy()

            self.plot_sepnet_pos(data_path, heatmap)

        elif self.mode == 'test':
            if data_path != None: data_list = self.get_image_list(data_path)
            else: data_list = self.test_dataset.images

            res_list = []

            for sample_batched in self.test_loader:
                sample_batched = sample_batched.cuda() if self.use_cuda else sample_batched
                heatmaps = self.sepposnet.forward(sample_batched)
                for j in range(heatmaps.shape[0]):
                    heatmap = heatmaps[j].detach().cpu().numpy()
                    res_list.append(heatmap)

            for d, res in zip(data_list, res_list):
                self.plot_sepnet_pos(d, res)


    # def plot_sepnet_pos(self, img_path, heatmaps):
    #     """
    #     plot setnet-p results for one image
    #     input: img_path [str], heatmaps: [2,H,W], 2 for number of keypoints
    #     """
    #     img = cv2.resize(cv2.imread(img_path), (self.img_w, self.img_h))
    #     splited = list(os.path.split(img_path))
    #     splited[-1] = 'out_' + splited[-1]
    #     splited.insert(-1, 'pred')
        
    #     if not os.path.exists(os.path.join(*splited[:-1])): 
    #         os.mkdir(os.path.join(*splited[:-1]))
    #     save_path = os.path.join(*splited)
        

    #     keypoints, overlays = [], []
    #     for h in heatmaps: 
    #         pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
    #         keypoints.append([pred_x, pred_y])
    #         vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #         vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    #         overlay = cv2.addWeighted(img, 0.7, vis, 0.3, 0)
    #         overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
    #         overlays.append(overlay)
    #     cv2.putText(overlays[0], 'pull', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    #     cv2.putText(overlays[1], 'hold', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    #     # cv2.imwrite(save_path, cv2.hconcat(overlays))
    #     cv2.imshow('sepnet-p result', cv2.hconcat(overlays))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    def infer_sep_dir(self, data_path=None, itvl=16):
        """
        infer picknet: check success/fail of the directions
        """
        if data_path != None and not os.path.exists(data_path):
            print("Invalid path! ")
            return
        # elif data_path != None and data_path.split('.')[-1] == 'png':
        elif self.mode == 'test':
            if data_path != None: data_list = self.get_image_list(data_path)
            else: data_list = self.test_dataset.images
            for d in data_list:
                img = cv2.resize(cv2.imread(d), (self.img_w, self.img_h))
                # infer a single image
                grasps = self.get_grasps_for_sepnet(img)
                
                heatmap = gauss_2d_batch(self.img_w, self.img_h, 8, grasps)
                img_t = torch.cat((self.transform(img), heatmap), 0)
                img_t = torch.unsqueeze(img_t, 0).cuda() if self.use_cuda else torch.unsqueeze(img_t, 0)
                scores = []
                for r in range(itvl):
                    direction = direction2vector(r*(360/itvl))
                    direction =  torch.from_numpy(direction).cuda() if self.use_cuda else torch.from_numpy(direction)
                    dir_t = direction.view(-1, direction.shape[0])
                    lbl_pred= self.sepdirnet.forward(img_t.float(), dir_t.float())
                    lbl_pred = torch.nn.Softmax(dim=1)(lbl_pred)
                    lbl_pred = lbl_pred.detach().cpu().numpy()
                    scores.append(lbl_pred.ravel()[1]) # only success possibility
                self.plot_sepnet_dir(d, scores, grasps)
                break

        elif self.mode == 'val':
            if data_path != None: data_list = self.get_image_list(data_path)
            else: data_list = self.val_dataset.images
            
            num_success = 0
            for sample_batched in self.val_loader:
                sample_batched = [Variable(d.cuda() if self.use_cuda else d) for d in sample_batched]
                img_t, dir_gt, labels_gt = sample_batched
                labels_pred = self.sepdirnet.forward(img_t.float(), dir_gt.float())
                for j in range(labels_pred.shape[0]):
                    lbl_pred = labels_pred[j].view(-1, labels_pred[j].shape[0])
                    lbl_pred = torch.nn.Softmax(dim=1)(lbl_pred)
                    lbl_gt = labels_gt[j]
                    if lbl_pred.argmax(dim=1)[0] == lbl_gt: num_success += 1
            
            print(f"Accuracy: {num_success}/{len(data_list)}", )

    # def plot_sepnet_dir(self, img_path, scores, grasps):
    #     """
    #     direction: first point to right, then counter-clockerwise
    #     """
    #     pull_p = grasps[0]
    #     hold_p = grasps[1]
    #     img = cv2.resize(cv2.imread(img_path), (self.img_w, self.img_h))
    #     splited = list(os.path.split(img_path))
    #     splited[-1] = 'out_' + splited[-1]
    #     splited.insert(-1, 'pred')
        
    #     if not os.path.exists(os.path.join(*splited[:-1])): 
    #         os.mkdir(os.path.join(*splited[:-1]))
    #     save_path = os.path.join(*splited)

    #     drawn = draw_vectors_bundle(img, start_p=pull_p, scores=scores)
    #     drawn = cv2.circle(drawn, pull_p,5,(0,255,0),-1)
    #     drawn = cv2.circle(drawn, hold_p,5,(0,255,0),-1)
        
    #     print("final score: ", np.round(scores, 3))
    #     cv2.imwrite(save_path, drawn)
    #     cv2.imshow('sepnet-d result', drawn)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    def plot(self, img_path, predictions, grasps=None):
        """
        img_path [str], results, grasps [list]
        """
        img = cv2.resize(cv2.imread(img_path), (self.img_w, self.img_h))

        splited = list(os.path.split(img_path))
        splited[-1] = 'out_' + splited[-1]
        splited.insert(-1, 'pred')

        if not os.path.exists(os.path.join(*splited[:-1])): 
            os.mkdir(os.path.join(*splited[:-1]))
        save_path = os.path.join(*splited)

        if self.infer_type == 'pick':
            scores, overlays = [], []
            for h in predictions: 
                pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                scores.append(h.max())
                vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img, 0.7, vis, 0.3, 0)
                overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
                overlays.append(overlay)
            
            if scores[0] > scores[1]:
                cv2.putText(overlays[0], 'pick', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (48, 192, 19), 2)
                cv2.putText(overlays[1], 'sep', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                cv2.putText(overlays[0], 'pick', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(overlays[1], 'sep', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (48, 192, 19), 2)

        elif self.infer_type == 'sep_pos':
            keypoints, overlays = [], []
            for h in predictions: 
                pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                keypoints.append([pred_x, pred_y])
                vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img, 0.7, vis, 0.3, 0)
                overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
                overlays.append(overlay)
            cv2.putText(overlays[0], 'pull', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(overlays[1], 'hold', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        elif self.infer_type == 'sep_dir':
            pull_p = grasps[0]
            hold_p = grasps[1]
            overlays = draw_vectors_bundle(img, start_p=pull_p, scores=predictions)
            overlays = cv2.circle(overlays, pull_p,5,(0,255,0),-1)
            overlays = cv2.circle(overlays, hold_p,5,(0,255,0),-1)
            
            print("final score: ", np.round(predictions, 3))

        # cv2.imwrite(save_path, cv2.hconcat(overlays))
        cv2.imshow(f'{self.infer_type} prediction', cv2.hconcat(overlays))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':

    config = InferConfig()
    inference = Inference(config=config)

    a = "C:\\Users\\xinyi\\Desktop\\tt.png"

    root_dir = "C:\\Users\\xinyi\\Documents\\Checkpoints\\try_SR\\model_epoch_"
    model_path = "C:\\Users\\xinyi\\Documents\\Checkpoints\\try_SR\\model_epoch_99.pth"
    # inference.infer_sep_pos(a)
    # inference.infer_sep_dir(a)

    # inference.infer_sep_pos()


    # for i in range(100):
    #     model_path = root_dir + str(i) + ".pth"
    #     # print(model_path)
    #     inference.infer_sep_dir_val(model_path)
    # inference.infer_sep_dir()
    # inference.infer_sep_dir_val(model_path)