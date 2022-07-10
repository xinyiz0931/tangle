"""
Trainer class
Author: xinyi
Date: 20220517
"""
import os
import logging
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tangle.utils import *
from tangle import PickNet, SepPositionNet, SepDirectionNet
from tangle import PickDataset, SepDataset

class Trainer(object):
    def __init__(self, config):

        self.config = config

        self.net_type = config.net_type
        self.epochs = config.epochs
        self.use_cuda = config.use_cuda
        
        self.out_dir = config.save_dir
        print(config.save_dir)
        self.timestamp_start = datetime.datetime.now()

        self.train_flag = True
        self.epoch = 0
        self.iteration = 0

        config.display()
        img_h = config.img_height
        img_w = config.img_width
        data_num = len(os.listdir(os.path.join(config.data_dir, 'images')))
        train_inds, test_inds = split_random_inds(data_num, config.test_ratio)
        # train_inds, test_inds = split_random_inds(1000,0.1)

        if self.net_type == 'pick':
            self.model = PickNet(model_type='unet', out_channels=2)
            self.criterion = torch.nn.MSELoss()
            self.optim = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate,
                         momentum=config.momentum, weight_decay=config.weight_decay)
            train_dataset = PickDataset(img_h, img_w, config.data_dir, train_inds)
            test_dataset = PickDataset(img_h, img_w, config.data_dir, test_inds)
            
        elif self.net_type == 'sep_pos':

            self.model = SepPositionNet(out_channels=2)

            self.criterion = torch.nn.BCELoss()
            self.optim = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, 
                         weight_decay=config.weight_decay)
            train_dataset = SepDataset(img_h, img_w, config.data_dir, self.net_type, data_inds=train_inds)
            test_dataset = SepDataset(img_h, img_w, config.data_dir, self.net_type, data_inds=test_inds)

        elif self.net_type == 'sep_dir':
            self.model = SepDirectionNet(in_channels=5, backbone='resnet')
            self.criterion = nn.CrossEntropyLoss()
            #self.optim = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, 
                        #  weight_decay=config.weight_decay)
            self.optim = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate, 
                         weight_decay=config.weight_decay)
            train_dataset = SepDataset(img_h, img_w, config.data_dir, self.net_type, data_inds=train_inds)
            test_dataset = SepDataset(img_h, img_w, config.data_dir, self.net_type, data_inds=test_inds)

        else:
            print('No such model type! Select from pick/sep_pos/sep_dir ... ')
            self.train_flag = False
            return
        
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.use_cuda: 
            self.model.to(self.device)

        # output and log, if save_dir exists, exit
        if os.path.exists(self.out_dir): 
            print('Oops! Save directory already exists ... ')
            answer = input("Do you want to re-write? [yes|no]: ").lower() 
            if answer != 'yes':
                self.train_flag = False
                return 
        else: os.mkdir(self.out_dir)
        self.log_headers = ['epoch','iteration','train_loss','test_loss','elapsed_time',]
        with open(os.path.join(self.out_dir, 'log.csv'), 'w') as f:
            f.write(','.join(self.log_headers) + '\n')
        config.record(os.path.join(self.out_dir, 'config.txt'))
        
    def forward(self, sample_batched):

        sample_batched = [d.to(self.device) for d in sample_batched]
        if self.net_type == 'pick':

            img, mask_gt = sample_batched
            img, mask_gt = sample_batched[0].to(self.device), sample_batched[1].to(self.device)
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

        elif self.net_type == 'test':
            inputs, labels = sample_batched
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
        return loss

    def train_epoch(self):

        train_loss = 0.0

        for i_batch, sample_batched in enumerate(self.train_loader):
            iteration = i_batch + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue
            self.iteration = iteration

            self.optim.zero_grad()
            loss = self.forward(sample_batched)

            loss.backward(retain_graph=True)
            self.optim.step()
            train_loss += loss.item()

            print('[%d, %5d/%d] loss: %.6f' % (self.epoch + 1, i_batch + 1, len(self.train_loader), loss.item()), end='')
            print('\r', end='')

            with open(os.path.join(self.out_dir, 'log.csv'), 'a') as f:
                elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.item()] + [''] + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

        print(f'Epoch {self.epoch+1} ==> train loss : {train_loss/i_batch}')

    def test_epoch(self):

        test_loss = 0.0

        for i_batch, sample_batched in enumerate(self.test_loader):
            loss = self.forward(sample_batched)
            test_loss += loss.item()
        print(f'Epoch {self.epoch+1} ==> test loss : {test_loss/i_batch}')
            

        with open(os.path.join(self.out_dir, 'log.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] + [test_loss / i_batch] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        torch.save(self.model.state_dict(), os.path.join(self.out_dir, f'model_epoch_{str(self.epoch)}.pth'))

    def train(self):
        # for epoch in tqdm.trange(self.epoch, self.epochs, desc='Train', ncols=100):
        for epoch in range(self.epochs):
            if self.train_flag: # to prevent overwriting files 
                self.epoch = epoch
                self.train_epoch()
                self.test_epoch()
