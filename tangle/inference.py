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
from tangle import PickNet, SepNet, SepNetD 
from tangle import PickDataset, SepDataset

class Inference(object):

    def __init__(self, config):
        
        self.config = config
        self.use_cuda = config.use_cuda
        self.mode = config.mode
        self.net_type = config.net_type
        self.sep_type = config.sep_type

        # config.display()

        self.img_h = config.img_height
        self.img_w = config.img_width
        self.batch_size = config.batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        data_list = []

        # check modelds" existence [pick, sep_pos, sep_dir]
        models = [False,False,False]
        
        if "pick" in self.net_type:
            self.picknet = PickNet(model_type="unet", out_channels=2)
            # self.picknet = torch.hub.load("pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=False)
            models[0] = True

        if "sep" in self.net_type:
            if self.net_type == "sep_pos":
                self.sepposnet = SepNet(out_channels=2)
                models[1] = True
            
            elif self.net_type == "sep_dir":
                if self.sep_type == "vector":
                    self.sepdirnet = SepNetD(in_channels=5, backbone="conv")
                elif self.sep_type == "spatial":
                    self.sepdirnet = SepNet(in_channels=4, out_channels=1)
                models[2] = True
            
            else:
                self.sepposnet = SepNet(out_channels=2)
                if self.sep_type == "vector": 
                    self.sepdirnet = SepNetD(in_channels=5, backbone="conv")
                else: 
                    self.sepdirnet = SepNet(in_channels=4, out_channels=1)
                
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
        self.exist_models = models
        
        # if validation mode, it"s necessary to load the dataset
        if self.mode == "val":
            # inds = random_inds(2, 100)
            # inds = random_inds(10,len(os.listdir(os.path.join(config.dataset_dir, "images"))))
            if "pick" in self.net_type:
                # self.val_dataset = PickDataset(self.img_h, self.img_w, config.dataset_dir, data_inds=inds) 
                self.val_dataset = PickDataset(self.img_h, self.img_w, config.dataset_dir) 
            elif "sep" in self.net_type:
                # self.val_dataset = SepDataset(self.img_h, self.img_w, config.dataset_dir, config.net_type, data_inds=inds)
                self.val_dataset = SepDataset(self.img_h, self.img_w, config.dataset_dir, config.net_type)
            self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size, shuffle=False)
        
        elif self.mode == "test":
            self.dataset_dir = config.dataset_dir
    
    def click(self, img, n=2):
        drawn = img.copy()
        points = [] # 2x2, [[pull_x, pull_y], [hold-x, hold_y], ...(n)]
        def on_click(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(drawn,(x,y),5,(0,255,0),-1)
                print(f"[*] {x},{y}")
                points.append([x,y])

        cv2.namedWindow("click twice to select pull and hold")
        cv2.setMouseCallback("click twice to select pull and hold",on_click)
        while(len(points)<n):
            cv2.imshow("click twice to select pull and hold",drawn)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or k==ord("q"):
                break
        cv2.destroyAllWindows()
        return np.array(points)

    def get_image_list(self, data_dir):
        """
        if data_dir == *.png: detect for one image
        else data_dir == directory: detect images in it
        return: a list contains all image paths 
        """
        data_list = []
        if os.path.isdir(data_dir) and os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.split(".")[-1] == "png": data_list.append(os.path.join(data_dir, f))
        elif os.path.isfile(data_dir) and os.path.exists(data_dir):
            if data_dir.split(".")[-1] == "png": 
                data_list.append(data_dir)
        else:
            print("[!] Invalid! Data path does not exist ")
        return data_list

    def infer_pick(self, data_list=None, _s=0):
        """Use PickNet to infer N samples

        Args:
            data_dir (str, optional): path to one file. Defaults to None.
            _s (int, optional): bounding size for collision avoiding. Defaults to 0.

        Returns:
            pick_or_sep (list): N x (0->pick/1->sep)
            pick_sep_p (list): N x (2x2)
            heatmaps (list) : N x (2xHxW)
        """

        heatmaps = []
        pick_or_sep = []
        pick_sep_p = []

        if self.mode == "test":

            for d in data_list:
                src_img = cv2.imread(d)
                src_h, src_w, _ = src_img.shape
                img = cv2.resize(src_img, (self.img_w, self.img_h))
                
                img_t = self.transform(img)
                img_t = torch.unsqueeze(img_t, 0).cuda() if self.use_cuda else torch.unsqueeze(img_t, 0)
                h = self.picknet(img_t)[0]
                h = h.detach().cpu().numpy()

                h_pick = cv2.resize(h[0], (src_w, src_h))
                h_sep = cv2.resize(h[1], (src_w, src_h))
                
                h_pick = cv2.rectangle(h_pick,(0,0),(src_w, src_h),(0,0,0),_s*2)
                h_sep = cv2.rectangle(h_sep,(0,0),(src_w, src_h),(0,0,0),_s*2)
                
                pick_y, pick_x = np.unravel_index(h_pick.argmax(), h_pick.shape)
                sep_y, sep_x = np.unravel_index(h_sep.argmax(), h_sep.shape)

                # pick_y, pick_x = np.unravel_index(h[0].argmax(), h[0].shape)
                # sep_y, sep_x = np.unravel_index(h[1].argmax(), h[1].shape)

                pick_sep_p.append(np.array([[pick_x, pick_y], [sep_x, sep_y]]))
                
                if h[0].max() > h[1].max(): pick_or_sep.append(0)
                else: pick_or_sep.append(1)
                heatmaps.append(np.array([h_pick, h_sep])) # 2xHxW
                
        elif self.mode == "val":
            for sample_batched in self.val_loader:
                sample_batched = [Variable(d.cuda() if self.use_cuda else d) for d in sample_batched]
                img, mask_gt = sample_batched
                heatmaps = self.picknet(img)
                mask_gt = mask_gt[0]
                # get ground truth label
                if mask_gt[0].max() >=  mask_gt[1].max(): lbl_gt = 0
                else: lbl_gt = 1

                for j in range(heatmaps.shape[0]):
                    if mask_gt[j][0].max() >=  mask_gt[j][1].max(): lbl_gt = 0
                    else: lbl_gt = 1

                    h = heatmaps[j].detach().cpu().numpy()
                    if h[0].max() >= h[1].max(): lbl_pred = 0
                    else: lbl_pred = 1
                    outputs.append([lbl_gt, lbl_pred])

        return pick_or_sep, pick_sep_p, heatmaps 

    def infer_sep_pos(self, data_list=None):
        """Use SepNet-P to infer pull point and hold point for N sampels

        Args:
            data_dir (str, optional): path to one image. Defaults to None.

        Returns:
            pull_hold_p (list): N x (2x2) 
            heatmaps (list): N x (2xHxW)
        """
        pull_hold_p = [] # pull + hold
        heatmaps = []
        
        if self.mode == "test":

            for img_path in data_list:
                src_img = cv2.imread(img_path)
                src_h, src_w, _ = src_img.shape
                
                img = cv2.resize(src_img, (self.img_w, self.img_h))
                img_t = self.transform(img)
                img_t = torch.unsqueeze(img_t, 0).cuda() if self.use_cuda else torch.unsqueeze(img_t, 0)
                heatmap = self.sepposnet.forward(img_t)[0] # (2,H,W)
                h = heatmap.detach().cpu().numpy()
                pullmap = cv2.resize(h[0], (src_w, src_h))
                holdmap = cv2.resize(h[1], (src_w, src_h))
                pull_y, pull_x = np.unravel_index(pullmap.argmax(), pullmap.shape)
                hold_y, hold_x = np.unravel_index(holdmap.argmax(), holdmap.shape)
                pull_hold_p.append(np.array([[pull_x, pull_y],[hold_x, hold_y]]))
                heatmaps.append(np.array([pullmap, holdmap]))
            
        return pull_hold_p, heatmaps

    def infer_sep_dir(self, data_list=None, pos=None, itvl=8):
        """Use SepNet-D to infer scores of `itvl` directions for N samples

        Args:
            data_dir (str, optional): path to one image. Defaults to None.
            pos (list, optional): Nx(2x2), pull and hold points. Defaults to None.
            itvl (int, optional): _description_. Defaults to 8.

        Returns:
            pull_hold_p (list): pull and hold points
            pull_v (list): pull vector
            - scores (list): scores of `itvl` directions when sep_type == "vector"    
            - heatmaps (list): Nx(itvlxHxW) when sep_type == "spatial"
        """
        if self.mode == "test": 

            scores = [] # scores for all directions
            pull_hold_p = []
            heatmaps = []
            pull_v = []
            for i, d in enumerate(data_list):

                src = cv2.imread(d)
                src_h, src_w, _ = src.shape
                rsz = cv2.resize(src, (self.img_w, self.img_h))
                
                if self.sep_type == "spatial":
                    # p_ : for original image before resizing
                    if pos == None: _p = self.click(src, n=1)
                    else: _p = pos[i]
                    # _p = np.array([[359, 182]])
                    if len(_p) != 1: return

                    # p: for resized image
                    p_hold = _p[0].copy()
                    p_hold[0] *= self.img_w / src_w
                    p_hold[1] *= self.img_h / src_h

                    pullmaps = []
                    max_score = -1
                    score_d = []
                    for j in range(itvl):
                        img = rsz.copy()
                        r = 360/itvl*j
                        img_r = rotate_img(img, r)
                        p_hold_r = rotate_pixel(p_hold, r, self.img_h, self.img_w)
                        img_r_t = self.transform(img_r)
                        img_r_t = torch.unsqueeze(img_r_t, 0).cuda() if self.use_cuda else torch.unsqueeze(img_r_t, 0)

                        holdmap_r_t = gauss_2d_batch(self.img_w, self.img_h, 8, np.array([p_hold_r])).float()
                        holdmap_r_t = torch.unsqueeze(holdmap_r_t, 0).cuda() if self.use_cuda else torch.unsqueeze(holdmap_r_t, 0)
                        inp_t = torch.cat((img_r_t, holdmap_r_t), 1)

                        pullmap_r_t = self.sepdirnet.forward(inp_t)[0][0] # (H,W)

                        pullmap_r = pullmap_r_t.detach().cpu().numpy()
                        _pullmap_r = cv2.resize(pullmap_r, (src_w, src_h))
                        pullmap_v = visualize_tensor(pullmap_r_t, cmap=True)
                        
                        y, x = np.unravel_index(pullmap_r.argmax(), pullmap_r.shape)
                        
                        p_pull = rotate_pixel((x,y), -r, self.img_h, self.img_w)
                        _p_pull = p_pull.copy()
                        _p_pull[0] *= src_w / self.img_w
                        _p_pull[1] *= src_h / self.img_h
                        vis_r = cv2.addWeighted(img_r, 0.65, pullmap_v, 0.35, -1)
                        cv2.circle(vis_r, (x,y), 7, (0,255,0), -1)
                        vis_r = draw_vector(vis_r, (x,y), (1,0), color=(0,255,255))
                        
                        if pullmap_r.max() > max_score:
                            max_score = pullmap_r.max()
                            _max_p = np.array([_p_pull])
                            _max_v = angle2vector(r)

                        score_d.append(pullmap_r.max())
                        # pullmaps.append(pullmap_r)
                        pullmaps.append(_pullmap_r)
                    _pos = np.vstack((_max_p, _p))

                    heatmaps.append(np.array(pullmaps)) 
                    pull_hold_p.append(_pos)
                    pull_v.append(_max_v)

                elif self.sep_type == "vector": 
                    if pos == None: p_ = self.click(src, n=2)
                    else: p_ = pos[i]
                    
                    if len(p_) != 2: return
                    else: pull_hold_p.append(p_)
                    
                    p = p_.copy()
                    p[:,0] = p_[:,0] * self.img_w / src_w
                    p[:,1] = p_[:,1] * self.img_h / src_h
                    heatmap = gauss_2d_batch(self.img_w, self.img_h, 8, p)
                    img_t = torch.cat((self.transform(rsz), heatmap), 0)
                    img_t = torch.unsqueeze(img_t, 0).cuda() if self.use_cuda else torch.unsqueeze(img_t, 0)
                    score = []
                    max_score = -1
                    for r in range(itvl):
                        vector = angle2vector(r*(360/itvl))
                        vector_t =  torch.from_numpy(vector).cuda() if self.use_cuda else torch.from_numpy(vector)
                        vector_t = vector.view(-1, vector_t.shape[0])
                        lbl_pred= self.sepdirnet.forward((img_t.float(), vector_t.float()))
                        lbl_pred = torch.nn.Softmax(dim=1)(lbl_pred)
                        lbl_pred = lbl_pred.detach().cpu().numpy()
                        score.append(lbl_pred.ravel()[1]) # only success possibility
                        if lbl_pred.ravel()[1] > max_score: 
                            max_v= vector 
                            max_score = lbl_pred.ravel()[1]
                    
                    scores.append(score)
                    pull_v.append(max_v)

            if self.sep_type == "spatial":  
                return pull_hold_p, pull_v, heatmaps
            elif self.sep_type == "vector": 
                return pull_hold_p, pull_v, scores

        # elif self.mode == "val":
            
        #     num_success = 0
        #     for sample_batched in self.val_loader:
        #         sample_batched = [Variable(d.cuda() if self.use_cuda else d) for d in sample_batched]
        #         img_t, dir_gt, labels_gt = sample_batched
        #         labels_pred = self.sepdirnet.forward(img_t.float(), dir_gt.float())
        #         for j in range(labels_pred.shape[0]):
        #             lbl_pred = labels_pred[j].view(-1, labels_pred[j].shape[0])
        #             lbl_pred = torch.nn.Softmax(dim=1)(lbl_pred)
        #             lbl_gt = labels_gt[j]
        #             if lbl_pred.argmax(dim=1)[0] == lbl_gt: num_success += 1
            
        #     print(f"[*] Accuracy: {num_success}/{len(self.val_loader)}")

        #     return pull_hold_p, scores

    def plot(self, img_path, predictions, sep_pos=None, sep_dir=None, show=False, save_dir=None):
        """
        img_path [str], results, grasps [list]
        """
        img = cv2.imread(img_path)
        
        s_out = list(os.path.split(img_path))
        s_ret = list(os.path.split(img_path))
        s_out[-1] = "out_" + s_out[-1]
        s_ret[-1] = "ret_" + s_ret[-1]
        s_out.insert(-1, "pred")
        s_ret.insert(-1, "pred")

        if save_dir == None:
            if not os.path.exists(os.path.join(*s_out[:-1])): 
                os.mkdir(os.path.join(*s_out[:-1]))
            save_out_path = os.path.join(*s_out)
            if not os.path.exists(os.path.join(*s_ret[:-1])): 
                os.mkdir(os.path.join(*s_ret[:-1]))
            save_ret_path = os.path.join(*s_ret)
        else: 
            save_out_path = os.path.join(save_dir, s_out[-1])
            save_ret_path = os.path.join(save_dir, s_ret[-1])

        # plot PickNet
        if sep_pos is None and sep_dir is None:

            scores, points, overlays = [], [], []
            for h in predictions[0:2]:
                scores.append(h.max())
                pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                points.append([pred_x, pred_y])
                vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img, 0.7, vis, 0.3, 0)
                overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
                overlays.append(overlay)
           
            out = cv2.hconcat(overlays)
            ret = img.copy()
            if scores[0] > scores[1]: 
                ret  = cv2.circle(ret, points[0], 7, (0, 255, 0), -1)
                cv2.putText(ret, "pick", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                ret  = cv2.circle(ret, points[1], 7, (0, 255, 0), -1)
                cv2.putText(ret, "sep", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        elif sep_pos is not None:

            pull_p, hold_p = sep_pos
            ret = img.copy()
            ret = cv2.circle(ret, pull_p,7,(0,255,0),-1)
            ret = cv2.circle(ret, hold_p,7,(0,255,255),2)
            
            if sep_dir is None: 
                preds_pick, preds_dir = None, None
                preds_pos = predictions

            else:
                if len(predictions) == 2: 
                    preds_pick = None
                    [preds_pos, preds_dir] = predictions
                elif len(predictions) == 3: 
                    [preds_pick, preds_pos, preds_dir] = predictions
                else:
                    preds_pick, preds_pos = None, None
                    preds_dir = predictions
                    
                
                ret = draw_vector(ret, pull_p, sep_dir, color=(0,255,0))

            out = []
            # --------- pick or sep ---------
            if preds_pick is not None: 
                overlays = []
                for h in preds_pick[0:2]:
                    pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                    vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img, 0.7, vis, 0.3, 0)
                    overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
                    overlays.append(overlay)

                out.append(cv2.vconcat(overlays))

            # --------- position ---------
            if preds_pos is not None: 
                overlays = []
                for h in preds_pos: 
                    pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                    vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img, 0.7, vis, 0.3, 0)
                    # overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
                    overlays.append(overlay)
                out.append(cv2.vconcat([overlays[1], overlays[0]]))

            # --------- direction --------- 
            if preds_dir is not None: 

                if self.sep_type == "spatial":  
                    n_col = 4
                    itvl = len(preds_dir)
                    overlays, rot_imgs, scores = [], [], []
                    for i in range(itvl):
                        r = 360 / itvl * i
                        img_r = rotate_img(img, r)
                        h = preds_dir[i]
                        scores.append(h.max())
                        pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                        heatmap = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(img_r, 0.65, heatmap, 0.35, -1)
                        cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
                        cv2.putText(overlay, str(np.round(h.max(), 3)), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        rot_imgs.append(overlay)

                    for i in np.arange(0, itvl, n_col):
                        overlays.append(cv2.hconcat(rot_imgs[i:i+n_col]))
                    out.append(cv2.vconcat(overlays))

                elif self.sep_type == "vector": 

                    all = draw_vectors_bundle(img.copy(), start_p=pull_p, scores=preds_dir)

                    out.append(cv2.vconcat((all, all)))
             
            out = cv2.hconcat(out)
        # print(f"[*] Save the heatmaps to {save_out_path}")
        print(f"[*] Save the results to {save_ret_path}")
        cv2.imwrite(save_out_path, out)
        cv2.imwrite(save_ret_path, ret)
        if show: 
            
            cv2.namedWindow(f"{self.net_type} heatmaps", cv2.WINDOW_NORMAL)
            cv2.imshow(f"{self.net_type} prediction", ret)
            cv2.imshow(f"{self.net_type} heatmaps", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def infer(self, data_dir=None, save=True, show=False, save_dir=None, net_type=None):
        """Infer use PickNet or SepNet

        Args:
            data_dir (str, optional): default to self.data_dir depending on net_type test/val.
            save (bool, optional): whether save the visualizaed heatmaps. Defaults to True.
            save_dir (str, optional): where to save visualized heatmaps. Defaults to data_dir/preds/out_*.
            net_type (_type_, optional): loaded 'sep' => net_type can be 'sep_pos' or 'sep_dir' 
        Returns:
            "pick"        : pick_sep_p, pn_scores
            "sep_pos"     : pull_hold_p 
            "sep_dir"     : pull_hold_p, pull_v, snd_scores
            "sep"         : pull_hold_p, pull_v, snd_scores
            "pick_sep"    : pick_or_sep, pick_sep_p, pn_scores, pull_hold_p, pull_v, snd_scores 
        """
        
        if data_dir != None: 
            data_list = self.get_image_list(data_dir)
        else: 
            data_list = self.get_image_list(self.dataset_dir)
        
        infer_flag = True
        if net_type is not None and net_type != self.net_type:
            if "pick" in net_type and not self.exist_models[0]:
                infer_flag = False
            if "sep" in net_type and (not self.exist_models[1] or not self.exist_models[2]):
                infer_flag = False
            if "pos" in net_type and not self.exist_models[1]:
                infer_flag = False
            if "dir" in net_type and not self.exist_models[2]:
                infer_flag = False
        if infer_flag == False: 
            print(f"[!] Existing models cannot support infer type: {net_type}")
            return
        
        if net_type is None: 
            net_type = self.net_type
         
        print(f"[*] Infer type: {net_type}")
        if net_type == "pick":
            # return three lists for N samples: 
            # (0) Pick or sep:       N x (0->pick/1->sep)
            # (1) PickNet positions: N x (2x2)
            # (2) PickNet scores:    N x (2)

            pick_or_sep, pick_sep_p, pn_heatmaps = self.infer_pick(data_list=data_list)
            if self.mode == "val":
                succ = 0
                for o in pn_heatmaps: 
                    if o[0] == o[1]: succ +=1 
                print("Success rate: ", succ," / ", len(pn_heatmaps))  
                return [succ, len(pn_heatmaps)]
            else:
                pn_scores = []
                for d, o in zip(data_list, pn_heatmaps):
                    pn_scores.append(np.array([o[0].max(), o[1].max()]))
                    if save: self.plot(d, o, save_dir=save_dir,show=show)
                return pick_or_sep, pick_sep_p, pn_scores

        elif net_type == "sep_pos":
            # return one listst for N sampels: 
            # (0) SepNet-P positions: N x (2x2) 
            pull_hold_p, snp_heatmaps = self.infer_sep_pos(data_list=data_list)
            for d, spo, p in zip(data_list, snp_heatmaps, pull_hold_p): 
                if save: self.plot(d, spo, sep_pos=p, save_dir=save_dir, show=show)
            return pull_hold_p

        elif net_type == "sep_dir":
            # return thress lists for N samples: 
            # (0) manually positions:  N x (2x2)
            # (1) SepNet-D directions: N x (2) 
            # (2) SepNet-D scores:     N x itvl_num

            pull_hold_p, pull_v, snd_outputs = self.infer_sep_dir(data_list=data_list)
            snd_scores = []

            for d, sdo, p, v in zip(data_list, snd_outputs, pull_hold_p, pull_v):
                snd_scores.append(np.array([s_.max() for s_ in sdo]))
                if save: self.plot(d, sdo, sep_pos=p, sep_dir=v, show=show, save_dir=save_dir)
            return pull_hold_p, pull_v, snd_scores

        elif net_type == "sep":
            # return three lists for N samples
            # (0) SepNet-P positions:  N x (2x2)
            # (1) SepNet-D directions: N x (2) 
            # (2) SepNet-D scores:     N x itvl_num
            
            snd_scores = []
            pos, snp_heatmaps = self.infer_sep_pos(data_list=data_list)
            hold_p = [p[1:2] for p in pos]
            if self.sep_type == "spatial": 
                pull_hold_p, pull_v, snd_outputs = self.infer_sep_dir(data_list=data_list, pos=hold_p)
            elif self.sep_type == "vector":
                pull_hold_p, pull_v, snd_outputs = self.infer_sep_dir(data_list=data_list, pos=pos, itvl=16)

            for d, spo, sdo, p, v in zip(data_list, snp_heatmaps, snd_outputs, pull_hold_p, pull_v):
                snd_scores.append(np.array([s_.max() for s_ in sdo]))
                if save: self.plot(d, [spo, sdo], sep_pos=p, sep_dir=v, save_dir=save_dir, show=show)
            return pull_hold_p, pull_v, snd_scores

        elif net_type == "pick_sep":
            # return three lists for N samples:
            # (0) Pick or sep:         N x (0->pick/1->sep)
            # (1) PickNet positions:   N x (2x2)
            # (2) PickNet scores:      N x (2)
            # (3) SepNet-P positions:  N x (2x2)
            # (4) SepNet-D directions: N x (2) 
            # (5) SepNet-D scores:     N x itvl_num

            pick_or_sep, pick_sep_p, pn_heatmaps = self.infer_pick(data_list=data_list)
            pick_idx = [i for i, s in enumerate(pick_or_sep) if s==0] 
            
            sep_idx = [i for i, s in enumerate(pick_or_sep) if s==1] 
            sep_data_list = [data_list[i] for i in sep_idx] 

            pos, snp_heatmaps = self.infer_sep_pos(data_list=sep_data_list)
            hold_p = [p[1:2] for p in pos]
            pull_hold_p, pull_v, snd_outputs = self.infer_sep_dir(data_list=sep_data_list, pos=hold_p)

            pn_scores_ = []
            pull_hold_p_, pull_v_, snd_scores_ = [], [], []
            for i in range(len(data_list)):
                pn_scores_.append(np.array([h_.max() for h_ in pn_heatmaps[i]]))
                if i in pick_idx: 
                    pull_hold_p_.append(None)
                    pull_v_.append(None)
                    snd_scores_.append(None)
                    if save: 
                        self.plot(data_list[i], pn_heatmaps[i], save_dir=save_dir,show=show)
                else: 
                    j = sep_idx.index(i)
                    pull_hold_p_.append(pull_hold_p[j])
                    pull_v_.append(pull_v[j])
                    snd_scores_.append(np.array([h_.max() for h_ in snd_outputs[j]]))
                    if save: 
                        self.plot(data_list[i], [pn_heatmaps[i], snp_heatmaps[j], snd_outputs[j]], 
                                  sep_pos=pull_hold_p[j], sep_dir=pull_v[j], save_dir=save_dir, show=show)

            return pick_or_sep, pick_sep_p, pn_scores_, pull_hold_p_, pull_v_, snd_scores_
           
        else: 
            print(f"Wrong infer type! ")

if __name__ == "__main__":

    from tangle import Config
    cfg = Config(config_type="infer")
    cfg.display()
    inference = Inference(config=cfg)
    
    # folder = "D:\\dataset\\picknet\\test\\depth0.png"
    # folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEightAugment\\images\\000161.png"
    # saved = "C:\\Users\\xinyi\\Desktop"
    # print(inference.get_image_list(folder))
    # folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEight\\images\\000004.png" 
    # folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEightAugment\\images\\000055.png" 
    # folder = "C:\\Users\\xinyi\\Documents\\Code\\bpbot\\data\\test\\depth20.png" 
    # folder = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes\\SC\\97\\depth.png" 
    # folder = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes\\SC\\35\\depth.png" 
    # folder = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes\\U\\122\\depth.png" 

    # res = inference.infer(data_dir=folder, net_type="pick")
    # res = inference.infer(data_dir=folder, save_dir=saved, net_type="sep_pos")
    # print(res)
    
    folder = "/home/hlab/Desktop/predicting/tmp5.png"
    
    saved = "/home/hlab/Desktop"
    output = inference.infer(data_dir=folder, net_type="sep", save_dir=saved, show=True)
    for f in output:
        print(f)
    # p, s = inference.infer(data_dir=folder, save_dir=saved,net_type="sep")
    # print(s)
