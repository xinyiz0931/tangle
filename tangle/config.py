"""
Configuration class
Author: xinyi
Date: 20220517
"""
from logging import root
import os

class InferConfig(object):
    # choice = ['test', 'val']
    mode = 'test'
    # choice = ['pick', 'sep', 'sep_pos', 'sep_dir', 'pick_sep']
    infer_type = 'pick'
    
    use_cuda = True
    batch_size = 1
    input_size = (512,512)

    root_dir = "C:\\Users\\xinyi\\Documents"
    root_dir = "C:\\Users\\matsumura\\Documents"

    pick_ckpt = os.path.join(root_dir, 'Checkpoints', 'try8', 'model_epoch_10.pth')
    sep_pos_ckpt = os.path.join(root_dir, 'Checkpoints', 'try_38', 'model_epoch_2.pth')
    sep_dir_ckpt = os.path.join(root_dir, 'Checkpoints', 'try_SR', 'model_epoch_99.pth')

    if mode == 'test':
        if 'pick' in infer_type:
            # dataset_dir = 'D:\\datasets\\picknet_dataset\\test'
            dataset_dir = os.path.join(root_dir, 'Datasets', 'picknet_dataset', 'test')
        elif 'sep' in infer_type:
            # dataset_dir = 'D:\\datasets\\sepnet_dataset\\test'
            dataset_dir = os.path.join(root_dir, 'Datasets', 'sepnet_dataset', 'test')
        else: 
            print(f"Wrong inference type: {infer_type} ... ")

    elif mode == 'val':
        if infer_type == 'pick':
            # dataset_dir = 'D:\\datasets\\picknet_dataset\\val'
            dataset_dir = os.path.join(root_dir, 'Datasets', 'picknet_dataset', 'val')

        elif infer_type == 'sep_pos' or infer_type == 'sep_dir':
            # dataset_dir = 'D:\\datasets\\sepnet_dataset\\val'
            dataset_dir = os.path.join(root_dir, 'Datasets', 'sepnet_dataset', 'val')
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