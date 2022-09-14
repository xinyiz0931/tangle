import os
import glob
import numpy as np
import torch
from torchvision import transforms
from tangle.utils import *
from tangle import PickNet, SepNet, SepNetD
from tangle import PickDataset, SepDataset

class Validation(object):
    def __init__(self, config):
        self.config = config
        self.net = config.net # pick, sep_hold, sep_pull, sep_dir (old)
        self.use_cuda = config.use_cuda

        config.display()
        self.img_h = config.img_height
        self.img_w = config.img_width
        self.transform = transforms.Compose([transforms.ToTensor()])

        inds = random_inds(10,100)
        
        if self.net == "pick":
            self.model = PickNet(model_type="unet", out_channels=2)
            self.ckpt_list = glob.glob(os.path.join(config.pick_ckpt, "*.pth")) 
            self.dataset = PickDataset(self.img_w, self.img_h, config.pick_dataset, data_inds=inds)
        
        elif self.net == "sep_hold":
            self.model = SepNet(out_channels=2)
            self.ckpt_list = glob.glob(os.path.join(config.sephold_ckpt, "*.pth"))
            self.dataset = SepDataset(self.img_w, self.img_h, config.sep_dataset, self.net, data_inds=inds)

        elif self.net == "sep_pull":
            self.model = SepNet(in_channels=4, out_channels=1)
            self.ckpt_list = glob.glob(os.path.join(config.seppull_ckpt, "*.pth")) 
            self.dataset = SepDataset(self.img_w, self.img_h, config.sep_dataset, self.net, data_type="val", data_inds=inds)
        
        elif self.net == "sep_dir":
            self.model = SepNetD(in_channels=5, backbone="conv")
            self.ckpt_list = glob.glob(os.path.join(config.sepdir_ckpt, "*.pth")) 
            self.dataset = SepDataset(self.img_w, self.img_h, config.sep_dataset, self.net, data_inds=inds)

        self.ckpt_list.sort(key=os.path.getmtime)

        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            print("[!] Invalid net input! ")
    
    def validate(self):
        if self.net == "sep_dir":
            for ckpt in self.ckpt_list[4:8]:
                if self.use_cuda:
                    self.model.load_state_dict(torch.load(ckpt)) 
                else:
                    self.model.load_state_dict(torch.load(ckpt), map_location=torch.device("cpu"))
                n_success = 0
                for data in self.dataset:
                    img, direction, lbl_gt = data
                    img_t = torch.unsqueeze(img, 0).cuda() if self.use_cuda else torch.unsqueeze(img, 0) 
                    direction_t = torch.unsqueeze(direction, 0).cuda() if self.use_cuda else torch.unsqueeze(direction, 0) 
                    lbl_pred = self.model.forward((img_t.float(), direction_t.float()))
                    lbl_pred = torch.nn.Softmax(dim=1)(lbl_pred)
                    if lbl_pred.argmax(dim=1)[0] == lbl_gt:
                        n_success += 1
                print(f"[*] {ckpt} accuracy: {n_success} / {len(self.dataset)} = {n_success/len(self.dataset)*100}%")


from tangle import Config
cfg = Config(config_type="validate")
val = Validation(config=cfg)
val.validate()
