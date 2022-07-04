"""
Configuration class: config_type = ['train', 'infer']
Author: xinyi
Date: 20220517
"""
from logging import root
import os
from chainer import config
import yaml

class Config(object):
    # choice : ['train', 'infer', 'serve']
    def __init__(self, config_data, config_type):
        self.config_type = config_type

        if isinstance(config_data, str): data = self.load(config_data)

        else: data = config_data
        root_dir = data['root_dir']
        pick_data_folder = data['pick_data_folder']
        sep_data_folder = data['sep_data_folder']

        try:
            self.config = data[self.config_type]
        except KeyError:
            raise AttributeError

        if self.config_type == 'train':
            self.save_dir = os.path.join(root_dir, 'Checkpoint', self.save_folder)
            if self.net_type == 'pick': 
                self.data_dir = os.path.join(root_dir, "Dataset", pick_data_folder, 'train')
            elif 'sep' in self.net_type:
                # data_folder = os.path.join(root_dir, "Dataset", "SepDataShape", "SR")
                # self.data_folder = "C:\\Users\\xinyi\\Documents\\Dataset\\HoldAndPullDirectionDataAll"
                self.data_dir = os.path.join(root_dir, 'Dataset', sep_data_folder, 'train')
            else:
                print("Wrong net type! Select from pick/sep_pos/sep_dir ...")

        elif self.config_type == 'infer':
            self.pick_ckpt = os.path.join(root_dir, 'Checkpoint', *self.pick_ckpt_folder)
            self.sepp_ckpt = os.path.join(root_dir, 'Checkpoint', *self.sepp_ckpt_folder)
            self.sepd_ckpt = os.path.join(root_dir, 'Checkpoint', *self.sepd_ckpt_folder)
            # self.pick_ckpt = os.path.join(root_dir, 'Checkpoints', 'try8', 'model_epoch_10.pth')
            # self.sep_pos_ckpt = os.path.join(root_dir, 'Checkpoints', 'try_38', 'model_epoch_4.pth')
            # self.sep_dir_ckpt = os.path.join(root_dir, 'Checkpoints', 'try_SR', 'model_epoch_50.pth')

            if self.mode == 'test':
                if 'pick' in self.infer_type:
                    self.dataset_dir = os.path.join(root_dir, 'Dataset', pick_data_folder, 'test')
                    # self.dataset_dir = 'D:\\datasets\\picknet_dataset\\test'
                    # dataset_dir = os.path.join(root_dir, 'Datasets', 'picknet_dataset', 'test')
                elif 'sep' in self.infer_type:
                    self.dataset_dir = os.path.join(root_dir, 'Dataset', sep_data_folder, 'test')
                    # self.dataset_dir = 'D:\\datasets\\sepnet_dataset\\test'
                    # dataset_dir = os.path.join(root_dir, 'Datasets', 'sepnet_dataset', 'test')
                else:  
                    print(f"Wrong inference type: {self.infer_type} ... ")
            elif self.mode == 'val':
                if self.infer_type == 'pick':
                    # self.dataset_dir = 'D:\\datasets\\picknet_dataset\\val'
                    # dataset_dir = os.path.join(root_dir, 'Datasets', 'picknet_dataset', 'val')
                    self.dataset_dir = os.path.join(root_dir, 'dataset', pick_data_folder, 'val')
                elif self.infer_type == 'sep_pos' or self.infer_type == 'sep_dir':
                    # self.dataset_dir = 'D:\\datasets\\sepnet_dataset\\val'
                    self.dataset_dir = os.path.join(root_dir, 'dataset', sep_data_folder, 'val')
                    # dataset_dir = os.path.join(root_dir, 'Datasets', 'sepnet_dataset', 'val')
                else:
                    print(f"Wrong mode/inference combination: {self.mode}/{self.infer_type} ... ")
            else:
                print(f"Wrong mode: {self.mode} ... ")
        

    def load(self, config_path):
        try:
            with open(config_path) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)

        except FileNotFoundError:
            print('Wrong file or file path')
        return data

    def __getattr__(self, key):
        try:
            return self.config[key]
        except KeyError:
            raise AttributeError
            
    def display(self):
        for a in self.config.keys():
            print("{:30} {}".format(a, self.config[a]))
        for a in dir(self):
            
            if not a.startswith("__") and not callable(getattr(self, a)) and not isinstance(getattr(self, a), dict):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def record(self, log_path):
        with open(log_path, "w") as f:
            for a in dir(self):
                if not a.startswith("__") and not callable(getattr(self, a)) and not isinstance(getattr(self, a), dict):
                    print("{:30} {}".format(a, getattr(self, a)), file=f)
            print("\n")
