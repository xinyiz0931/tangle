import os
import sys
import logging
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import tqdm
from tangle.utils import *
from tangle import PickNet, SepPositionNet, SepDirectionNet
from tangle import PickDataset, SepDataset

class TrainConfig(object):
    # choice = ['pick', 'sep_pos', 'sep_dir']
    net_type = 'sep_pos'

    epochs = 100
    use_cuda = True
    use_cuda_infer = False
    device = 'cuda:0'
    input_size = (512,512)
    batch_size = 1
    learning_rate = 1e-4
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 0 
    test_ratio = 0.15

    root_dir = "/home/xpredictioninyi/Documents/"
    root_dir = "C:\\Users\\xinyi\\Documents"

    # save_folder = os.path.join(root_dir, "Checkpoints", "try_SR_no_hold")
    save_folder = os.path.join(root_dir, "Checkpoints", "try_38_t")
    
    # data folder for PickNet
    if net_type == 'pick': 
        data_folder = os.path.join(root_dir, "Dataset", "PickData")
    elif 'sep' in net_type:
        # data_folder = os.path.join(root_dir, "Dataset", "SepDataShape", "SR")
        data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\HoldAndPullDirectionDataAll"
    else: 
        print("Wrong net type! Select from pick/sep_pos/sep_dir ...")

    def __init__(self):
        pass
    
    def display(self):
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def record(self, log_path):
        with open(log_path, "w") as f:
            for a in dir(self):
                if not a.startswith("__") and not callable(getattr(self, a)):
                    print("{:30} {}".format(a, getattr(self, a)), file=f)
            print("\n")

class Trainer(object):
    def __init__(self, config):

        self.config = config

        self.net_type = config.net_type
        self.epochs = config.epochs
        self.use_cuda = config.use_cuda
        
        self.out_dir = config.save_folder
        self.timestamp_start = datetime.datetime.now()
        
        

        self.epoch = 0
        self.iteration = 0

        config.display()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        (img_h, img_w) = config.input_size
        data_num = len(os.listdir(os.path.join(config.data_folder, 'images')))
        train_inds, test_inds = split_random_inds(data_num, config.test_ratio)

        if self.net_type == 'pick':
            self.model = PickNet(model_type='unet', out_channels=2)
            self.criterion = nn.MSELoss()
            self.optim = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate,
                         momentum=config.momentum, weight_decay=config.weight_decay)
            train_dataset = PickDataset(img_h, img_w, config.data_folder, train_inds)
            test_dataset = PickDataset(img_h, img_w, config.data_folder, test_inds)
            
        elif self.net_type == 'sep_pos':
            self.model = SepPositionNet(out_channels=2)

            # add fine-tuning
            pretrained_dict = torch.load("C:\\Users\\xinyi\\Documents\\Checkpoints\\try_38\\model_epoch_7.pth")
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

            self.criterion = nn.BCELoss()
            self.optim = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, 
                         weight_decay=config.weight_decay)
            train_dataset = SepDataset(img_h, img_w, config.data_folder, self.net_type, data_inds=train_inds)
            test_dataset = SepDataset(img_h, img_w, config.data_folder, self.net_type, data_inds=test_inds)

        elif self.net_type == 'sep_dir':
            self.model = SepDirectionNet(in_channels=4)
            self.criterion = nn.CrossEntropyLoss()
            self.optim = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, 
                         weight_decay=config.weight_decay)
            train_dataset = SepDataset(img_h, img_w, config.data_folder, self.net_type, data_inds=train_inds)
            test_dataset = SepDataset(img_h, img_w, config.data_folder, self.net_type, data_inds=test_inds)
        else:
            print('No such model type! Select from pick/sep_pos/sep_dir ... ')
            return
        
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        
        if self.use_cuda: 
            # self.model = self.model.cuda()
            self.model.to(device)

        # output and log, if save_dir exists, exit
        if os.path.exists(self.out_dir): 
            print('Oops! Save directory already exists ... ')
            answer = input("Do you want to re-write? [yes|no]: ") 
            if answer == "no": 
                self.train_flag = False
                return 
            else: self.train_flag = True
        else:
            self.train_flag = True
            os.mkdir(self.out_dir)

        self.log_headers = ['epoch','iteration','train_loss','test_loss','elapsed_time',]
        with open(os.path.join(self.out_dir, 'log.csv'), 'w') as f:
            f.write(','.join(self.log_headers) + '\n')
        config.record(os.path.join(self.out_dir, 'config.txt'))
        
    def forward(self, sample_batched):

        sample_batched = [Variable(d.cuda() if self.use_cuda else d) for d in sample_batched]
        
        if self.net_type == 'pick':
            img, mask_gt = sample_batched
            mask_pred = self.model(img.float())
            loss = self.criterion(mask_pred, mask_gt.float())

        elif self.net_type == 'sep_pos':
            img, gauss_gt = sample_batched
            gauss_pred = self.model.forward(img).double()
            loss = self.criterion(gauss_pred, gauss_gt)

        elif self.net_type == 'sep_dir': 
            img, direction, lbl_gt = sample_batched
            lbl_pred = self.model.forward(img.float(), direction.float())
            logging.debug(f"pred: {lbl_pred.shape}, gt: {lbl_gt.shape}")
            loss = self.criterion(lbl_pred, lbl_gt.long())

        return loss

    def train_epoch(self):

        train_bar = tqdm.tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc='Train epoch=%d' % self.epoch,
            ncols=100,
            leave=False,
        )

        train_loss = 0.0

        for i_batch, sample_batched in train_bar:
            iteration = i_batch + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue
            self.iteration = iteration

            self.optim.zero_grad()
            loss = self.forward(sample_batched)

            loss.backward(retain_graph=True)
            self.optim.step()
            train_loss += loss.item()

            train_bar.set_postfix_str(f'Loss:{loss.item():.6f}')

            with open(os.path.join(self.out_dir, 'log.csv'), 'a') as f:
                elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.item()] + [''] + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

        #print('\ntrain loss:',  train_loss / i_batch)

    def test_epoch(self):
        test_bar = tqdm.tqdm(
            enumerate(self.test_loader),
            total=len(self.test_loader),
            desc='Test epoch=%d' % self.epoch,
            ncols=100,
            leave=False,
        )

        test_loss = 0.0
        for i_batch, sample_batched in test_bar:
            loss = self.forward(sample_batched)
            test_loss += loss.item()
            test_bar.set_postfix_str(f'Loss:{loss.item():.6f}')

        with open(os.path.join(self.out_dir, 'log.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] + [test_loss / i_batch] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        torch.save(self.model.state_dict(), os.path.join(self.out_dir, f'model_epoch_{str(self.epoch)}.pth'))

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.epochs, desc='Train', ncols=100):
            if self.train_flag: # to solve the write problem
                self.epoch = epoch
                self.train_epoch()
                self.test_epoch()
