import os
import glob
import numpy as np
import torch
from torchvision import transforms
from tangle.utils import *
from tangle import PickNet, SepNet, SepNetD
from tangle import PickDataset, PullDataset

class Validation(object):
    def __init__(self, config):
        self.config = config
        self.net = config.net # pick, sep_hold, sep_pull, sep_dir (old)
        self.use_cuda = config.use_cuda

        config.display()
        self.img_h = config.img_height
        self.img_w = config.img_width
        self.transform = transforms.Compose([transforms.ToTensor()])
 
        if self.net == "pick":
            self.model = PickNet(model_type="unet", out_channels=2)
            self.dataset = PickDataset(self.img_w, self.img_h, config.dataset)
        
        elif self.net == "sep_hold":
            self.model = SepNet(out_channels=2)
            self.dataset = PullDataset(self.img_w, self.img_h, config.dataset, self.net)

        elif self.net == "sep_pull":
            self.model = SepNet(in_channels=4, out_channels=1)
            self.dataset = PullDataset(self.img_w, self.img_h, config.dataset, self.net, data_type="val")
        
        elif self.net == "sep_dir":
            self.model = SepNetD(in_channels=5, backbone="conv")
            # self.model = SepNetD(in_channels=5, backbone="resnet50")
            self.dataset = PullDataset(self.img_w, self.img_h, config.dataset, self.net)

        self.ckpt_list = glob.glob(os.path.join(config.ckpt_dir, "*.pth"))
        self.ckpt_list.sort(key=os.path.getmtime)

        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            print("[!] Invalid net input! ")
    
    def validate(self):

        if self.net == "sep_dir":
            
            log_headers = ["epoch", "acc", "acc_one"]
            with open(os.path.join(self.config.ckpt_dir, "acc.csv"), "w") as f:
                f.write(",".join(log_headers) + '\n')
            for i in np.arange(0, len(self.ckpt_list)):
                ckpt = self.ckpt_list[i]
                if self.use_cuda:
                    self.model.load_state_dict(torch.load(ckpt)) 
                else:
                    self.model.load_state_dict(torch.load(ckpt), map_location=torch.device("cpu"))
                
                n_success = 0
                n_one = 0
                n_one_success = 0

                for data in self.dataset:
                    img, direction, lbl_gt = data
                    img_t = torch.unsqueeze(img, 0).cuda() if self.use_cuda else torch.unsqueeze(img, 0) 
                    direction_t = torch.unsqueeze(direction, 0).cuda() if self.use_cuda else torch.unsqueeze(direction, 0) 
                    pred = self.model.forward((img_t.float(), direction_t.float()))
                    pred = torch.nn.Softmax(dim=1)(pred)
                    
                    lbl_pred = pred.argmax(dim=1)[0].cpu().numpy()
                    lbl_gt = lbl_gt.cpu().numpy()
                    # print("GT: ", lbl_gt, "Pred: ", lbl_pred, pred)
                    if lbl_gt == 1: 
                        n_one += 1
                        if lbl_pred == 1: n_one_success += 1
                    if lbl_pred == lbl_gt:
                        n_success += 1
                
                pctg_all = np.round(n_success/len(self.dataset), 3)
                pctg_one = np.round(n_one_success/n_one, 3)
                with open(os.path.join(self.config.ckpt_dir, "acc.csv"), "a") as f:
                    log = [i, pctg_all,pctg_one]
                    log = map(str, log)
                    f.write(",".join(log) + '\n')               
                
                print(f"[*] {os.path.split(ckpt)[-1]}: acc (all) = {n_success}/{len(self.dataset)} = {pctg_all}, acc (one) = {n_one_success}/{n_one} = {pctg_one}")

        elif self.net == "sep_pull":
            log_headers = ["epoch", "acc", "acc_one"]
            with open(os.path.join(self.config.ckpt_dir, "acc.csv"), "w") as f:
                f.write(",".join(log_headers) + '\n')
            from bpbot.utils import rotate_img, rotate_pixel
            
            for i in np.arange(0, len(self.ckpt_list), 1):
                ckpt = self.ckpt_list[i]

                if self.use_cuda:
                    self.model.load_state_dict(torch.load(ckpt))
                else:
                    self.model.load_state_dict(torch.load(ckpt), map_location=torch.device("cpu"))

                n_success = 0
                sum_cos_sim = 0
                for data in self.dataset:
                    img, position, lbl_gt = data
                    itvl = len(lbl_gt)
                    lbl_pred = []
                    for i in range(itvl):
                        img = img.copy()
                        r = 360/itvl * i
                        img_r = rotate_img(img, r)
                        img_r_t = self.transform(img_r)
                        p_hold_r = rotate_pixel(position[1], r, self.img_w, self.img_h)

                        holdmap_r_t = gauss_2d_batch(self.img_w, self.img_h, 8, [p_hold_r]).float()
                        
                        inp_t = torch.cat((img_r_t, holdmap_r_t), axis=0) 
                        inp_t = torch.unsqueeze(inp_t, 0).cuda() if self.use_cuda else torch.unsqueeze(inp_t, 0)

                        pullmap_r_t = self.model.forward(inp_t)[0][0]
                        pullmap_r = pullmap_r_t.detach().cpu().numpy()

                        y, x = np.unravel_index(pullmap_r.argmax(), pullmap_r.shape)
                        p_pull = rotate_pixel((x,y), -r, self.img_w, self.img_h)

                        lbl_pred.append(pullmap_r.max())
                    # coding for validation metric
                    lbl_categorized = np.zeros(itvl, dtype=int)
                    lbl_categorized[np.argmax(lbl_pred)] = 1

                    # scores: cosine similarity? 
                    # categorized: success or failure wherel = 1
                    cos_sim = np.dot(lbl_gt, lbl_pred) / (np.linalg.norm(lbl_gt) * np.linalg.norm(lbl_pred)) 
                    sum_cos_sim += cos_sim
                    if np.argmax(lbl_pred) in np.where(lbl_gt==1)[0]:
                        n_success += 1

                with open(os.path.join(self.config.ckpt_dir, "acc.csv"), "a") as f:
                    log = [i, n_success/len(self.dataset),sum_cos_sim/len(self.dataset)]
                    log = map(str, log)
                    f.write(",".join(log) + '\n')               

                print(f"[*] {os.path.split(ckpt)[-1]} : accuracy={n_success}/{len(self.dataset)}={n_success/len(self.dataset)*100}%, similarity={sum_cos_sim/len(self.dataset)}")
            
from tangle import Config
cfg = Config(config_type="validate")
val = Validation(config=cfg)
val.validate()
