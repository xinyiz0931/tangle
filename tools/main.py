
import argparse
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
from tangle import TrainConfig, Trainer

parser = argparse.ArgumentParser()

parser.add_argument('--mode', choices=['train','validate','infer'], help='functions')
parser.add_argument('--infer', choices=['pick', 'sep', 'sep_pos', 'sep_dir', 'pick_sep'])
parser.add_argument('--val', choices=['pick', 'sep_pos', 'sep_dir'])
def main():
    args = parser.parse_args()
    if args.mode == 'train':
        # train the model
        config = TrainConfig()
        trainer = Trainer(config=config)
        trainer.train()

    # elif args.mode == 'validate':
    #     # same as the training but iwht accuracy



if __name__ == '__main__':
    main()
    
