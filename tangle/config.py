"""
Configuration class
Author: xinyi
Date: 20220517
"""
from logging import root
import os

# class TrainConfig(object):
#     # choice = ['pick', 'sep_pos', 'sep_dir']
#     net_type = 'sep_dir'
    
#     epochs = 100
#     use_cuda = True
#     use_cuda_infer = False
#     device = 'cuda:0'
#     input_size = (512,512)
#     batch_size = 1
#     learning_rate = 1e-4
#     momentum = 0.9
#     weight_decay = 1e-4
#     num_workers = 0 
#     test_ratio = 0.15

#     root_dir = "/home/xpredictioninyi/Documents/"
#     root_dir = "C:\\Users\\xinyi\\Documents"

#     save_folder = os.path.join(root_dir, "Checkpoints", "try_SR")
    
#     # data folder for PickNet
#     if net_type == 'pick': 
#         data_folder = os.path.join(root_dir, "Dataset", "PickData")
#     elif 'sep' in net_type:
#         data_folder = os.path.join(root_dir, "Dataset", "SepDataShape", "SR")
#     else: 
#         print("Wrong net type! Select from pick/sep_pos/sep_dir ...")

#     def __init__(self):
#         pass
    
#     def display(self):
#         for a in dir(self):
#             if not a.startswith("__") and not callable(getattr(self, a)):
#                 print("{:30} {}".format(a, getattr(self, a)))
#         print("\n")

#     def record(self, log_path):
#         with open(log_path, "w") as f:
#             for a in dir(self):
#                 if not a.startswith("__") and not callable(getattr(self, a)):
#                     print("{:30} {}".format(a, getattr(self, a)), file=f)
#             print("\n")

# class InferConfig(object):
#     # choice = ['pick', 'sep_pos', 'sep_dir']
#     net_type = 'sep_dir'
    
#     epochs = 100
#     use_cuda = True
#     use_cuda_infer = False
#     device = 'cuda:0'
#     input_size = (512,512)
#     batch_size = 1
#     learning_rate = 1e-4
#     momentum = 0.9
#     weight_decay = 1e-4
#     num_workers = 0 
#     test_ratio = 0.15

#     root_dir = "/home/xpredictioninyi/Documents/"
#     root_dir = "C:\\Users\\xinyi\\Documents"

#     save_folder = os.path.join(root_dir, "Checkpoints", "try_SR")
    
#     # data folder for PickNet
#     if net_type == 'pick': 
#         data_folder = os.path.join(root_dir, "Dataset", "PickData")
#     elif 'sep' in net_type:
#         data_folder = os.path.join(root_dir, "Dataset", "SepDataShape", "SR")
#     else: 
#         print("Wrong net type! Select from pick/sep_pos/sep_dir ...")

#     def __init__(self):
#         pass
    
#     def display(self):
#         for a in dir(self):
#             if not a.startswith("__") and not callable(getattr(self, a)):
#                 print("{:30} {}".format(a, getattr(self, a)))
#         print("\n")

#     def record(self, log_path):
#         with open(log_path, "w") as f:
#             for a in dir(self):
#                 if not a.startswith("__") and not callable(getattr(self, a)):
#                     print("{:30} {}".format(a, getattr(self, a)), file=f)
#             print("\n")

