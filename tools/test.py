import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rotate
from tangle.utils import *
from bpbot.utils import rotate_img
from gen_data import augment_data
itvl = 8

def np_gauss_2d_batch(img_w, img_h, sigma, locs):
    locs = np.array(locs)
    X,Y = torch.meshgrid([torch.arange(0, img_w), torch.arange(0, img_h)])
    X = torch.transpose(X, 0, 1)
    Y = torch.transpose(Y, 0, 1)
    U = torch.from_numpy(locs[:,0])
    V = torch.from_numpy(locs[:,1])
    U.unsqueeze_(1).unsqueeze_(2)
    V.unsqueeze_(1).unsqueeze_(2)
    
    G = torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    
    U = np.expand_dims(locs[:,0], axis=(-1,-2))
    V = np.expand_dims(locs[:,1], axis=(-1,-2))
    X,Y = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h))
    G = np.exp(-((X-U)**2+(Y-V)**2)/(2.0*sigma**2))


import pandas as pd
p = "C:\\Users\\xinyi\\Documents\\Checkpoint\\try_action_map_augment_real\\acc.csv"
df = pd.read_csv(p)
df = df.drop(df.columns[[0]], axis=1)  
print(df)
df.to_csv(p, index=False)