import os
import timeit
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from bpbot.utils import *
from tangle.utils import *
from tangle import PickNet, SepNet

class Inference(object):

    def __init__(self, config):
        
        self.config = config
        self.use_cuda = config.use_cuda
        self.mode = config.mode
        self.net_type = config.net_type

        # config.display()

        self.img_h = config.img_height
        self.img_w = config.img_width
        self.batch_size = config.batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])

        
        if "pick" in self.net_type:
            # self.picknet = PickNet(model_type="unet", out_channels=3)
            self.picknet = PickNet(model_type="unet", out_channels=2)
            # self.picknet = torch.hub.load("pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=False)
            if self.use_cuda:
                self.picknet = self.picknet.cuda()
                self.picknet.load_state_dict(torch.load(config.picknet_ckpt))
            else:
                self.picknet.load_state_dict(torch.load(config.picknet_ckpt,map_location=torch.device("cpu")))  

        if "sep" in self.net_type:
            # ------------------ OLD ---------------------
            self.sepnet = SepNet(in_channels=3, out_channels=1)
            # ------------------ OLD ---------------------
            if self.use_cuda: 
                self.sepnet = self.sepnet.cuda()
                self.sepnet.load_state_dict(torch.load(config.sepnet_ckpt))
            else:
                self.sepnet.load_state_dict(torch.load(config.sepnet_ckpt,map_location=torch.device("cpu")))

        # if validation mode, it"s necessary to load the dataset
        # if self.mode == "val":
        #     # inds = random_inds(2, 100)
        #     # inds = random_inds(10,len(os.listdir(os.path.join(config.dataset_dir, "images"))))
        #     if "pick" in self.net_type:
        #         # self.val_dataset = PickDataset(self.img_h, self.img_w, config.dataset_dir, data_inds=inds) 
        #         self.val_dataset = PickDataset(self.img_w, self.img_h, config.dataset_dir) 
        #     elif "sep" in self.net_type:
        #         # self.val_dataset = SepDataset(self.img_h, self.img_w, config.dataset_dir, config.net_type, data_inds=inds)
        #         self.val_dataset = SepDataset(self.img_w, self.img_h, config.dataset_dir, config.net_type)
        #     self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # elif self.mode == "test":
        #     self.dataset_dir = config.dataset_dir

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

    def infer_pick(self, data_list, _s=0):
        """Use PickNet to infer N samples

        Args:
            data_list (list): list of file path. 
            _s (int, optional): bounding size for bin collision avoiding. defaults to 0.

        Returns:
            pick_sep_p (list): N x (2,2)
            scores (list):     N x (2,)
            heatmaps (list) :  N x (2,H,W)
        """

        heatmaps = []
        scores = []
        pick_sep_p = []

        if self.mode == "test":

            for d in data_list:
                print("[*] Infer image", d)
                src_img = cv2.imread(d)
                src_h, src_w, _ = src_img.shape
                img = cv2.resize(src_img, (self.img_w, self.img_h))
                
                img_t = self.transform(img)
                img_t = torch.unsqueeze(img_t, 0).cuda() if self.use_cuda else torch.unsqueeze(img_t, 0)
                h = self.picknet(img_t)[0]
                h = h.detach().cpu().numpy()
                d_p, d_scores, d_heatmaps = [], [], []
                for h_ in h:
                    heatmap = cv2.rectangle(h_, (0,0), (self.img_w, self.img_h), (0,0,0), _s*2)
                    y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
                    p = [x,y]
                    p[0] *= src_w / self.img_w
                    p[1] *= src_h / self.img_h     
                    d_p.append(p)
                    d_scores.append(heatmap.max())
                    d_heatmaps.append(heatmap)
                pick_sep_p.append(d_p)
                scores.append(d_scores)
                heatmaps.append(d_heatmaps)

        return np.array(pick_sep_p, dtype=int), np.array(scores), np.array(heatmaps)

        # elif self.mode == "val":
        #     for sample_batched in self.val_loader:
        #         sample_batched = [Variable(d.cuda() if self.use_cuda else d) for d in sample_batched]
        #         img, mask_gt = sample_batched
        #         heatmaps = self.picknet(img)
        #         mask_gt = mask_gt[0]
        #         # get ground truth label
        #         if mask_gt[0].max() >=  mask_gt[1].max(): lbl_gt = 0
        #         else: lbl_gt = 1

        #         for j in range(heatmaps.shape[0]):
        #             if mask_gt[j][0].max() >=  mask_gt[j][1].max(): lbl_gt = 0
        #             else: lbl_gt = 1

        #             h = heatmaps[j].detach().cpu().numpy()
        #             if h[0].max() >= h[1].max(): lbl_pred = 0
        #             else: lbl_pred = 1
        #             outputs.append([lbl_gt, lbl_pred])


    def infer_sep(self, data_list, itvl=8):
        """Use SepNet to infer N samples

        Args:
            data_list (list): list of file path. 
            itvl (int, optional): sampled directions of pulling. defaults to 8.
        Returns:
            pull_p (list):   N x (2,)
            pull_v (list):   N x (2,)
            scores (list):   N x (itvl,)
            heatmaps (list): N x (itvl,H,W)
        """

        pull_p, pull_v, scores, heatmaps = [], [], [], []
        
        for d in data_list:
            print("[*] Infer image", d)
            # tmp saved list for a single file `d`
            # d_p:        (2,) vector point to right
            # d_scores:   (itvl,)
            # d_heatmaps: (itvl,H,W)

            if d is None or d == "":
                pull_p.append(None)
                pull_v.append(None)
                scores.append(None)
                heatmaps.append(None)
                continue
                
            src = cv2.imread(d)
            src_h, src_w, _ = src.shape
            rsz = cv2.resize(src, (self.img_w, self.img_h))

            d_p, d_heatmaps, d_scores = [], [], []

            for j in range(itvl):
                print("tmp: ", j)
                img = rsz.copy()
                r = 360/itvl*j
                img_r = rotate_img(img, r)
                inp_r_t = torch.unsqueeze(self.transform(img_r), 0)
                inp_r_t = inp_r_t.cuda() if self.use_cuda else inp_r_t
                # ------------------------- OLD -----------------------
                pullmap_r_t = self.sepnet.forward(inp_r_t)[0][0]
                # ------------------------- NEW -----------------------
                # pullmap_r_t = self.sepnet(inp_r_t)['out'][0][0]
                pullmap_r = pullmap_r_t.detach().cpu().numpy()

                y, x = np.unravel_index(pullmap_r.argmax(), pullmap_r.shape)
                p = rotate_pixel((x,y), -r, self.img_w, self.img_h)
                _p = p.copy()
                _p[0] *= src_w / self.img_w
                _p[1] *= src_h / self.img_h

                # _pullmap_r = cv2.resize(pullmap_r, (src_w, src_h))
                # d_p.append(list(map(int, _p))) 
                d_p.append(_p)
                d_scores.append(pullmap_r.max())
                d_heatmaps.append(pullmap_r)

            d_scores = np.asarray(d_scores)
            maxid = d_scores.argmax() 
            # maxid = d_scores.index(max(d_scores))
            v = angle2vector(360/itvl*maxid)
            p = d_p[maxid]

            pull_p.append(p)
            pull_v.append(v)
            scores.append(d_scores)
            heatmaps.append(d_heatmaps)
        
        return pull_p, pull_v, scores, heatmaps

    def plot(self, img_path, preds, net_type=None, save_dir=None, show=False):
        """Plot heatmaps and scores

        Args:
            img_path (str): path to a single image
            preds (array or tuple): 
                "pick"     (2,H,W)
                "sep"      (itvl,H,W)
                "pick_sep" ((2,H,W),(itvl,H,W))
            net_type (str, optional): "pick"/"sep"/"pick_sep". Defaults to None.
            save_dir (str, optional): save directory. Defaults to None (path/to/img/preds).
            show (bool, optional): show the plotted heatmaps. Defaults to False. 
        """
        if net_type is None: net_type = self.net_type
        
        img = cv2.imread(img_path)
        rsz = cv2.resize(img, (self.img_w, self.img_h))
        s_out = list(os.path.split(img_path))
        s_ret = list(os.path.split(img_path))
        # s_out[-1] = "out_" + s_out[-1]
        s_out[-1] = net_type + "net_" + s_out[-1]
        # s_ret[-1] = "ret_" + s_ret[-1]
        s_out.insert(-1, "pred")
        # s_ret.insert(-1, "pred")

        if save_dir == None:
            if not os.path.exists(os.path.join(*s_out[:-1])): 
                os.mkdir(os.path.join(*s_out[:-1]))
            save_out_path = os.path.join(*s_out)
            if not os.path.exists(os.path.join(*s_ret[:-1])): 
                os.mkdir(os.path.join(*s_ret[:-1]))
            # save_ret_path = os.path.join(*s_ret)
        else: 
            save_out_path = os.path.join(save_dir, s_out[-1])
            # save_ret_path = os.path.join(save_dir, s_ret[-1])

        # plot PickNet: preds = (2xHxW)

        if net_type == "pick":
            scores, overlays = [], []
            for h,name in zip(preds, ["pick", "sep"]):
                scores.append(h.max())
                # pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                # points.append([pred_x, pred_y])
                vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(rsz, 0.5, vis, 0.5, 0)
                # cv2.putText(overlay, name+' '+str(np.round(h.max(), 3)), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                # overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
                overlays.append(overlay) 
            maxid = scores.index(max(scores))
            # cv2.rectangle(overlays[maxid], (0,0),(overlays[maxid].shape[1],overlays[maxid].shape[0]), (0,255,0),5)
            out = cv2.hconcat(overlays)
            
            # ret = img.copy()
            # if scores[0] > scores[1]: 
            #     ret  = cv2.circle(ret, points[0], 7, (0, 255, 0), -1)
            #     cv2.putText(ret, "pick", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            # else:
            #     ret  = cv2.circle(ret, points[1], 7, (0, 255, 0), -1)
            #     cv2.putText(ret, "sep", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        elif net_type == "sep":
            scores, overlays = [], []
            for j, h in enumerate(preds):
                r = 360 / len(preds) * j
                rot_rsz = rotate_img(rsz, r)
                scores.append(h.max())
                vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(rot_rsz, 0.5, vis, 0.5, 0)
                # cv2.putText(overlay, "pull "+str(np.round(h.max(), 3)), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                overlays.append(overlay) 
            maxid = scores.index(max(scores))
            cv2.rectangle(overlays[maxid], (0,0),(overlays[maxid].shape[1],overlays[maxid].shape[0]), (0,255,0),5)
            n_col = 4
            # n_col = 3
            # overlays.append(np.zeros_like(overlays[0], dtype=np.uint8))
            multi_overlays= []
            for c in np.arange(0,len(overlays),n_col):
                multi_overlays.append(cv2.hconcat(overlays[c:c+n_col]))
            out = cv2.vconcat(multi_overlays)
        
        elif net_type == "pick_sep":
            pn_preds, sn_preds = preds
            # picknet first
            scores, overlays = [], []
            for h in pn_preds:
                scores.append(h.max())
                # pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
                # points.append([pred_x, pred_y])
                vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(rsz, 0.65, vis, 0.35, 0)
                # cv2.putText(overlay, str(np.round(h.max(), 3)), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                # overlay = cv2.circle(overlay, (pred_x, pred_y), 7, (0, 255, 0), -1)
                overlays.append(overlay) 
            maxid = scores.index(max(scores))
            cv2.rectangle(overlays[maxid], (0,0),(overlays[maxid].shape[1],overlays[maxid].shape[0]), (0,255,0),5)
            pn_out = cv2.hconcat(overlays)
            # pn_ret = img.copy()
            # if scores[0] > scores[1]: 
            #     pn_ret  = cv2.circle(pn_ret, points[0], 7, (0, 255, 0), -1)
            #     cv2.putText(pn_ret, "pick", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            # else:
            #     pn_ret  = cv2.circle(pn_ret, points[1], 7, (0, 255, 0), -1)
            #     cv2.putText(pn_ret, "sep", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            # sepnet then
            scores, overlays = [], []
            for j, h in enumerate(sn_preds):
                r = 360 / len(sn_preds) * j
                rot_rsz = rotate_img(rsz, r)
                scores.append(h.max())
                vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(rot_rsz, 0.65, vis, 0.35, 0)
                # cv2.putText(overlay, str(np.round(h.max(), 3)), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                overlays.append(overlay) 
            maxid = scores.index(max(scores))
            cv2.rectangle(overlays[maxid], (0,0),(overlays[maxid].shape[1],overlays[maxid].shape[0]), (0,255,0),5)
            n_col = 4
            # adjust col number
            # n_col = 3
            # overlays.append(np.zeros_like(overlays[0], dtype=np.uint8))
            multi_overlays= []
            for c in np.arange(0,len(overlays),n_col):
                multi_overlays.append(cv2.hconcat(overlays[c:c+n_col]))
            sn_out = cv2.vconcat(multi_overlays)
            
            out = cv2.vconcat([pn_out, cv2.resize(sn_out, None, fx=0.5, fy=0.5)])
        print(f"[*] Saved in {save_out_path}")
        cv2.imwrite(save_out_path, out)
        if show: 
            cv2.imshow(net_type, out)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def infer(self, data_dir, net_type=None, save_dir=None, save=True, show=False):
        """Infer using PickNet or SepNet

        Args:
            data_dir (str): _description_
            net_type (str, optional): reloaded network type. Defaults to None.
            save_dir (str, optional): where to save visualized heatmaps. Defaults to None (data/dir/preds).
            save (bool, optional): if saving visualized heatmapes. Defaults to True.
            show (bool, optional): if showing visualized heatmaps. Defaults to False.

        Returns:
            "pick"    : pick_sep_p (list) : N x (2,2)
                        pn_scores  (list) : N x (2,)  
            "sep"     : pull_p     (list) : N x (2,)
                        pull_v     (list) : N x (2,)
                        sn_scores     (list) : N x (itvl,)
            "pick_sep": all above 
        """ 
        
        if data_dir != None: 
            data_list = self.get_image_list(data_dir)
        else: 
            data_list = self.get_image_list(self.dataset_dir)
        self.data_list = data_list

        if net_type is None: 
            net_type = self.net_type
         
        print(f"[*] Infer type: {net_type}")
        if net_type == "pick":
            self.return_keys = ["pick_or_sep?", "pick_p"]
            pick_sep_p, pn_scores, pn_heatmaps = self.infer_pick(data_list=data_list)
            pick_sep_cls = np.argmax(pn_scores, axis=1)
            pick_p = pick_sep_p[np.arange(len(pick_sep_cls)),pick_sep_cls,:]
            if self.mode == "val":
                succ = 0
                for o in pn_heatmaps: 
                    if o[0] == o[1]: succ +=1 
                print("Success rate: ", succ," / ", len(pn_heatmaps))  
                return [succ, len(pn_heatmaps)]
            else:
                for d, h in zip(data_list, pn_heatmaps):
                    if save: self.plot(img_path=d, preds=h, net_type=net_type, save_dir=save_dir, show=show)
                # return pick_sep_cls, pick_p 
                return pick_sep_p, pn_scores
        
        elif net_type == "sep":
            self.return_keys = ["pull_p", "pull_v"]
            pull_p, pull_v, sn_scores, sn_heatmaps = self.infer_sep(data_list=data_list)
            for d, h in zip(data_list, sn_heatmaps):
                if save:
                    self.plot(img_path=d, preds=h, net_type=net_type, save_dir=save_dir, show=show)
            return pull_p, pull_v

        elif net_type == "pick_sep":
            self.return_keys = ["pick_sep_cls", "pick_p", "pull_p", "pull_v"]
            pick_sep_p, pn_scores, pn_heatmaps = self.infer_pick(data_list=data_list)
            pick_sep_cls = np.argmax(pn_scores, axis=1)
            pick_p = pick_sep_p[np.arange(len(pick_sep_cls)),pick_sep_cls,:]
            sep_data_list = ["" if list(pick_sep_cls)[i]==0 else s for i,s in enumerate(data_list)]
     
            pull_p, pull_v, sn_scores, sn_heatmaps = self.infer_sep(data_list=sep_data_list)
            print("[*] SepNet scores: ", sn_scores)

            for i in range(len(data_list)):
                if sep_data_list[i] is None or sep_data_list[i] == "": 
                    if save: 
                        self.plot(data_list[i], pn_heatmaps[i], net_type="pick", save_dir=save_dir, show=show)
                else: 
                    if save: 
                        self.plot(data_list[i], (pn_heatmaps[i], sn_heatmaps[i]), save_dir=save_dir, show=show)

            return pick_sep_cls, pick_p, pull_p, pull_v
        else: 
            print(f"Wrong infer type! ")

if __name__ == "__main__":

    from tangle import Config
    cfg = Config(config_type="infer")
    cfg.display()
    inference = Inference(config=cfg)
    
    # folder = "D:\\dataset\\picknet\\test\\depth0.png"
    folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEightAugment\\images\\000161.png"
    folder = "C:\\Users\\xinyi\\Desktop\\_tmp\\012.png"
    # folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorVal\\SR"
    # folder = "C:\\Users\\xinyi\\Desktop\\_tmp"
    # folder = "C:\\Users\\xinyi\\Desktop\\val_image"
    # print(inference.get_image_list(folder))
    # folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEight\\images\\000004.png" 
    # folder = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEightAugment\\images\\000055.png" 
    # folder = "C:\\Users\\xinyi\\Code\\bpbot\\data\\test\\depth28.png" 
    # folder = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes\\SC\\97\\depth.png" 
    # folder = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes\\SC\\35\\depth.png" 
    # folder = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes\\U\\122\\depth.png" 

    # res = inference.infer(data_dir=folder, net_type="pick")
    # res = inference.infer(data_dir=folder, save_dir=saved, net_type="sep_pos")
    # print(res)
    
    # folder = "/home/hlab/Desktop/predicting/tmp5.png"
    
    # saved = "/home/hlab/Desktop"
    output = inference.infer(data_dir=folder, net_type="sep", show=True)
    # print(output)
    # keys = ["pull_p", "pull_v"]
    # for i, d in enumerate(inference.data_list):
    #     img = cv2.imread(d)
        
    #     print("--------", d, "pick or sep? ")
        # pick_sep_cls = output[0][i]
        # pick_p = output[1][i]
        # pull_p = output[2][i]
        # pull_v = output[3][i]
        # if pull_p is None: 
        #     cv2.circle(img, pick_p, 5, (0,255,0), -1)
        #     cv2.imshow("Action: Pick", img)
        # else: 
        #     cv2.circle(img, pull_p, 5, (0,255,0), -1)
        #     cv2.arrowedLine(img, pull_p, (pull_p+pull_v * 50).astype(int), (0,255,0), 2)
        #     cv2.imshow("Action: Sep", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # pull_p = output[2][i]
        # pull_v = output[3][i]
        # for k, out in zip(inference.return_keys,output):
        #     print("%14s: " % k, out[i].flatten() if out[i] is not None else "")
            # print("%14s: " % k, type(out[i]), out[i])

            