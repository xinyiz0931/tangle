import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from bpbot.utils import *
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

        if self.net == "pick":
            self.model = PickNet(model_type="unet", out_channels=2)
            self.ckpt_list = glob.glob(os.path.join(config.pick_ckpt, "*.pth")) 
        
        elif self.net == "sep_hold":
            self.model = SepNet(out_channels=2)
            self.ckpt_list = glob.glob(os.path.join(config.sephold_ckpt, "*.pth")) 

        elif self.net == "sep_pull":
            self.model = SepNet(in_channels=4, out_channels=1)
            self.ckpt_list = glob.glob(os.path.join(config.seppull_ckpt, "*.pth")) 
        
        elif self.net == "sep_dir":
            self.model = SepNetD(in_channels=5, backbone="conv")
            self.ckpt_list = glob.glob(os.path.join(config.sepdir_ckpt, "*.pth")) 

        self.ckpt_list.sort(key=os.path.getmtime)
        
        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            print("[!] Invalid net input! ")

from tangle import Config
cfg = Config(config_type="validate")
val = Validation(config=cfg)
