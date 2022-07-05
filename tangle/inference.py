import os
import timeit
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from bpbot.utils import *
from tangle.utils import *
from tangle import PickNet, SepPositionNet, SepDirectionNet
from tangle import PickDataset, SepDataset

class Inference(object):

    def __init__(self, config):
        
        self.config = config
        self.use_cuda = config.use_cuda
        self.mode = config.mode
        self.infer_type = config.infer_type
        # config.display()

        self.img_h = config.img_height
        self.img_w = config.img_width
        self.batch_size = config.batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.data_list = []

        # check modelds' existence [pick, sep_pos, sep_dir]
        models = [False,False,False]

        if 'pick' in self.infer_type:
            # self.picknet = PickNet(model_type='fcn', out_channels=2)
            self.picknet = torch.hub.load("pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=False)
            models[0] = True

        if 'sep' in self.infer_type:
            if self.infer_type == 'sep_pos':
                self.sepposnet = SepPositionNet(out_channels=2)
                models[1] = True

            elif self.infer_type == 'sep_dir':
                self.sepdirnet = SepDirectionNet(in_channels=5, backbone='resnet')
                models[2] = True
            else:
                self.sepposnet = SepPositionNet(out_channels=2)
                self.sepdirnet = SepDirectionNet(in_channels=5, backbone='resnet')
                models[1] = True
                models[2] = True
        
        if models[0] is True: 
            if self.use_cuda:
                self.picknet = self.picknet.cuda()
                self.picknet.load_state_dict(torch.load(config.pick_ckpt))
            else:
                self.picknet.load_state_dict(torch.load(config.pick_ckpt,map_location=torch.device("cpu")))  
        if models[1] is True:
            if self.use_cuda: 
                self.sepposnet = self.sepposnet.cuda()
                self.sepposnet.load_state_dict(torch.load(config.sepp_ckpt))
            else:
                self.sepposnet.load_state_dict(torch.load(config.sepp_ckpt,map_location=torch.device("cpu")))
        if models[2] is True:
            if self.use_cuda: 
                self.sepdirnet = self.sepdirnet.cuda()
                self.sepdirnet.load_state_dict(torch.load(config.sepd_ckpt))
            else:
                self.sepdirnet.load_state_dict(torch.load(config.sepd_ckpt,map_location=torch.device("cpu")))
            
        # if validation mode, it's necessary to load the dataset
        if self.mode == 'val':
            # inds = random_inds(2, 100)
            # inds = random_inds(10,len(os.listdir(os.path.join(config.dataset_dir, "images"))))
            if 'pick' in self.infer_type:
                # self.val_dataset = PickDataset(self.img_h, self.img_w, config.dataset_dir, data_inds=inds) 
                self.val_dataset = PickDataset(self.img_h, self.img_w, config.dataset_dir) 
            elif 'sep' in self.infer_type:
                # self.val_dataset = SepDataset(self.img_h, self.img_w, config.dataset_dir, config.infer_type, data_inds=inds)
                self.val_dataset = SepDataset(self.img_h, self.img_w, config.dataset_dir, config.infer_type)
            self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size, shuffle=False)
        
        elif self.mode == 'test':
            self.dataset_dir = config.dataset_dir

    def get_grasps_for_sepnet(self, img):
        drawn = img.copy()
        grasps = [] # 2x2, [[pull_x, pull_y], [hold-x, hold_y]]
        def on_click(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(drawn,(x,y),5,(0,255,0),-1)
                notice_print(f"{x},{y}")
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

    def get_image_list(self, data_dir):
        """
        if data_dir == *.png: detect for one image
        else data_dir == directory: detect images in it
        return: a list contains all image paths 
        """
        data_list = []
        if os.path.isdir(data_dir) and os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.split('.')[-1] == 'png': data_list.append(os.path.join(data_dir, f))
        elif os.path.isfile(data_dir) and os.path.exists(data_dir):
            if data_dir.split('.')[-1] == 'png': 
                data_list.append(data_dir)
        else:
            warning_print("Invalid! Data path does not exist ")
        return data_list

    def infer_pick(self, data_dir=None):
        """
        infer picknet: classify two maps: pick/sep maps
        return: action_type: [0,1]
                location: [[x1,y1]]
        """
        outputs = []
        pick_or_sep = []
        pick_sep_p = []

        if self.mode == 'test':
            if self.data_list == []:
                if data_dir != None: self.data_list = self.get_image_list(data_dir)
                else: self.data_list = self.get_image_list(self.dataset_dir)

            for d in self.data_list:
                img = cv2.resize(cv2.imread(d), (self.img_w, self.img_h))
                img_t = self.transform(img)
                img_t = torch.unsqueeze(img_t, 0).cuda() if self.use_cuda else torch.unsqueeze(img_t, 0)
                h = self.picknet(img_t)['out'][0]
                h = h.detach().cpu().numpy()

                pick_y, pick_x = np.unravel_index(h[0].argmax(), h[0].shape)
                sep_y, sep_x = np.unravel_index(h[1].argmax(), h[1].shape)

                pick_sep_p.append([[pick_x, pick_y], [sep_x, sep_y]])

                if h[0].max() > h[1].max(): pick_or_sep.append(0)
                else: pick_or_sep.append(1)
                outputs.append(h) # 2xHxW

        elif self.mode == 'val':
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
       
        return pick_or_sep, pick_sep_p, outputs 

    def infer_sep_pos(self, data_dir=None):
        """
        no validation mode
        infer sepnet-p: output two keypoints: hold, pull
        """
        pull_hold_p = []
        outputs = []

        if self.mode == 'test':

            if data_dir != None: self.data_list = self.get_image_list(data_dir)
            else: self.data_list = self.get_image_list(self.dataset_dir)

            for img_path in self.data_list:
                print("infer: ", img_path)
                img = cv2.resize(cv2.imread(img_path), (self.img_w, self.img_h))
                img_t = self.transform(img)
                img_t = torch.unsqueeze(img_t, 0).cuda() if self.use_cuda else torch.unsqueeze(img_t, 0)
                heatmap = self.sepposnet.forward(img_t)[0] # (2,H,W)
                h = heatmap.detach().cpu().numpy()

                pull_y, pull_x = np.unravel_index(h[0].argmax(), h[0].shape)
                hold_y, hold_x = np.unravel_index(h[1].argmax(), h[1].shape)
                pull_hold_p.append([[pull_x, pull_y],[hold_x, hold_y]])
                outputs.append(h)
            
        return pull_hold_p, outputs

    def infer_sep_dir(self, data_dir=None, grasps=None, itvl=16):
        """
        infer picknet: check success/fail of the directions
        """
        outputs = [] # scores for all directions
        pull_hold_p = []
        if self.mode == 'test': 
            # if self.data_list == []: 
            if data_dir != None: self.data_list = self.get_image_list(data_dir)
            else: self.data_list = self.get_image_list(self.dataset_dir)
            for i, d in enumerate(self.data_list):

                img = cv2.resize(cv2.imread(d), (self.img_w, self.img_h))
                # infer a single image
                if grasps == None: g_ = self.get_grasps_for_sepnet(img)
                else: g_ = grasps[i]
                
                if len(g_) != 2: return
                else: pull_hold_p.append(g_)
                heatmap_no_hold = gauss_2d_batch(self.img_w, self.img_h, 8, np.array([g_[0]])) 
                heatmap = gauss_2d_batch(self.img_w, self.img_h, 8, g_)
                img_t = torch.cat((self.transform(img), heatmap), 0)
                img_t = torch.unsqueeze(img_t, 0).cuda() if self.use_cuda else torch.unsqueeze(img_t, 0)
                score = []
                for r in range(itvl):
                    direction = direction2vector(r*(360/itvl))
                    direction =  torch.from_numpy(direction).cuda() if self.use_cuda else torch.from_numpy(direction)
                    dir_t = direction.view(-1, direction.shape[0])
                    lbl_pred= self.sepdirnet.forward(img_t.float(), dir_t.float())
                    lbl_pred = torch.nn.Softmax(dim=1)(lbl_pred)
                    lbl_pred = lbl_pred.detach().cpu().numpy()
                    score.append(lbl_pred.ravel()[1]) # only success possibility
                
                outputs.append(score)
                # self.plot(d, scores, grasps)
            if grasps != None: pull_hold_p = grasps

        elif self.mode == 'val':
            # if data_dir != None: data_list = self.get_image_list(data_dir)
            # else: data_list = self.val_dataset.images
            
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
            
            notice_print(f"Accuracy: {num_success}/{len(self.val_loader)}")

        return pull_hold_p, outputs

    def plot(self, img_path, predictions, show=False, grasps=None, cmap=True, plot_type=None, save_dir=None):
        """
        img_path [str], results, grasps [list]
        """
        if plot_type == None: plot_type = self.infer_type
        img = cv2.resize(cv2.imread(img_path), (self.img_w, self.img_h))
        
        splited = list(os.path.split(img_path))
        splited[-1] = 'out_' + splited[-1]
        splited.insert(-1, 'pred')

        if save_dir == None:
            if not os.path.exists(os.path.join(*splited[:-1])): 
                os.mkdir(os.path.join(*splited[:-1]))
            save_path = os.path.join(*splited)
        
        else: save_path = os.path.join(save_dir, splited[-1])

        if plot_type == 'pick':

            scores, overlays = [], []
            for h in predictions[0:2]:
                pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                scores.append(h.max())
                vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if cmap: vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                else: vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
                overlay = cv2.addWeighted(img, 0.7, vis, 0.3, 0)
                overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
                overlays.append(overlay)

            if cmap: 
                if scores[0] > scores[1]:
                    cv2.putText(overlays[0], 'pick', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (48, 192, 19), 2)
                    cv2.putText(overlays[1], 'sep', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                else:
                    cv2.putText(overlays[0], 'pick', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(overlays[1], 'sep', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (48, 192, 19), 2)
                showing = cv2.hconcat(overlays)
            else:
                if scores[0] > scores[1]: showing = overlays[0]
                else: showing = overlays[1]

        elif plot_type == 'sep_pos':
            points, overlays = [], []
            for h in predictions: 
                pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if cmap: 
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img, 0.7, vis, 0.3, 0)
                    overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
                    overlays.append(overlay)
                else: 
                    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
                    points.append([pred_x, pred_y])
            # ======= TEMP ========
            _tmp_l = list(os.path.split(img_path))
            _tmp_path = os.path.join("D:\\dataset\\sepnet\\train\\heatmaps", _tmp_l[-1])
            vis = predictions[1]
            vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) 
            cv2.imwrite(_tmp_path, vis)
            # =====================
             
            if cmap:
                cv2.putText(overlays[0], 'pull', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(overlays[1], 'hold', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                showing = cv2.hconcat([overlays[1], overlays[0]])
            else:
                showing = img.copy()
                showing = cv2.circle(showing, points[0], 7, (0, 255, 0), -1)
                showing = cv2.circle(showing, points[1], 7, (0, 0, 255), -1)

        elif plot_type == 'sep_dir':
            pull_p = grasps[0]
            hold_p = grasps[1]
            showing = draw_vectors_bundle(img, start_p=pull_p, scores=predictions)
            showing = cv2.circle(showing, pull_p,5,(0,255,0),-1)
            showing = cv2.circle(showing, hold_p,5,(0,255,0),-1)
            
            notice_print(f"final score: {np.round(predictions, 3)}")
        
        elif plot_type == 'sep':

            [predictions_pos, predictions_dir] = predictions

            pull_p = grasps[0]
            hold_p = grasps[1]
            showing = draw_vectors_bundle(img.copy(), start_p=pull_p, scores=predictions_dir)
            showing = cv2.circle(showing, pull_p,5,(0,255,0),-1)
            showing = cv2.circle(showing, hold_p,5,(0,255,0),-1)
            
            notice_print(f"final score: {np.round(predictions_dir, 3)}")
            if cmap: 
                points, overlays = [], []
                for h in predictions_pos: 
                    pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                    vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img.copy(), 0.7, vis, 0.3, 0)
                    overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
                    overlays.append(overlay)
                showing = cv2.hconcat([overlays[1], overlays[0], showing])
        
        main_proc_print(f"save the result to {save_path}")

        cv2.imwrite(save_path, showing)

        if show: 
            cv2.imshow(f'{self.infer_type} prediction', showing)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return showing

    def infer(self, cmap=True, data_dir=None, save_dir=None, save=True, infer_type=None):
        self.data_list = [] # ininitialize 
        if infer_type is None: infer_type = self.infer_type
        
        if infer_type == 'pick':
            pick_or_sep, pick_sep_p, outputs = self.infer_pick(data_dir=data_dir)
            if save: 
                for d, o in zip(self.data_list, outputs):
                    self.plot(d, o, cmap=cmap, save_dir=save_dir, plot_type=infer_type)
            return [pick_or_sep, pick_sep_p]

        elif infer_type == 'sep_pos':
            pull_hold_p, outputs = self.infer_sep_pos(data_dir=data_dir)
            if save: 
                for d, o in zip(self.data_list, outputs):
                    self.plot(d, o, cmap=cmap, show=False, save_dir=save_dir, plot_type=infer_type)
            return pull_hold_p

        elif infer_type == 'sep_dir':
            pull_hold_p, outputs = self.infer_sep_dir(data_dir=data_dir)
            if save: 
                for d, o, p in zip(self.data_list, outputs, pull_hold_p):
                    self.plot(d, o, grasps=p, cmap=cmap, save_dir=save_dir, plot_type=infer_type)
            return pull_hold_p

        elif infer_type == 'sep':
            pull_hold_p, outputs_pos = self.infer_sep_pos(data_dir=data_dir)
            
            _, outputs_dir = self.infer_sep_dir(data_dir=data_dir, grasps=pull_hold_p)
            if save:            
                for d, op, od, p in zip(self.data_list, outputs_pos, outputs_dir, pull_hold_p):
                    self.plot(d, [op, od], grasps=p, cmap=cmap, save_dir=save_dir, plot_type=infer_type)
            return [pull_hold_p, outputs_dir]
        
        elif infer_type == 'pick_sep_pos':
            pick_or_sep, pick_sep_p, outputs_pick = self.infer_pick(data_dir=data_dir)
            for d, o_pick, l in zip(self.data_list, outputs_pick, pick_or_sep):
                if l == 1:
                    p, o_sepp  = self.infer_sep_pos(data_dir=d) 
                    pull_hold_p = p
                    if save: self.plot(d, o_sepp[0], cmap=cmap, show=False, plot_type='sep_pos', save_dir=save_dir)
                else:
                    pull_hold_p = None
                    if save: self.plot(d, o_pick, cmap=cmap, plot_type='pick', save_dir=save_dir)
            return [pick_or_sep, pick_sep_p, pull_hold_p]

        elif infer_type == 'pick_sep':
            pick_or_sep, pick_sep_p, outputs_pick = self.infer_pick(data_dir=data_dir)
            for d, o_pick, l in zip(self.data_list, outputs_pick, pick_or_sep):
                if l == 1:
                    p, o_sepp  = self.infer_sep_pos(data_dir=d) 
                    _, o_sepd = self.infer_sep_dir(data_dir=d, grasps=p)
                    pull_hold_p = p
                    outputs_dir = o_sepd
                    if save: self.plot(d, [o_sepp[0],o_sepd[0]], grasps=p[0], plot_type='sep', save_dir=save_dir)
                else:
                    pull_hold_p, outputs_dir = None, None
                    if save: self.plot(d, o_pick, plot_type='pick', save_dir=save_dir)
            return [pick_or_sep, pick_sep_p, pull_hold_p, outputs_dir]
        else: 
            print(f"Wrong infer type! ")
if __name__ == '__main__':

    from tangle import Config
    config_path = "./cfg\\config.yaml"
    cfg = Config(config_data=config_path, config_type="infer")
    inference = Inference(config=cfg)
    
    folder = "D:\\dataset\\picknet\\test"
    # folder = "D:\\dataset\\sepnet\\val\\images"
    # folder = "C:\\Users\\xinyi\\Pictures"
    # print(inference.get_image_list(folder))
    
    res = inference.infer(data_dir=folder, infer_type='pick_sep')
    # res = inference.infer(data_dir=folder, save_dir="C:\\Users\\xinyi\\Desktop", infer_type='pick_sep_pos')
    print(res)
    # inference.infer(save_dir="C:\\Users\\xinyi\\Desktop")
