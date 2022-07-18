import os
import json
from xml.sax.handler import property_interning_dict
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tangle.utils import *
from bpbot.utils import *

# for pickdata: rotation-involved
seq_pick = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(
        scale=(0.8,1),
        rotate=[0,90,180,270],
        shear=(-10,10),
        ),
    #iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=0.025*255))
    ],random_order=True)

# for dragdata: no rotation-involved
seq_drag = iaa.Sequential([
    iaa.Flipud(0.5),
    iaa.Affine(
        scale=(0.8,1),
        shear=(-10,10)
        ),
    #iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=0.025*255)),
    iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=0.5, sigma=0.35))
    ],random_order=True)

def gen_sep_data(source_dir, dest_dir):
    """
    ├ dest_dir
    ├── images
    │   ├── 000000.png
    │   └── ...
    ├── positions.npz - np.array([[pull_x, pull_y, hold_x, hold_y], [...], ...]), shape = (16 x N)
    └── direction.npz - np.array([[1,0,0,0,...], [...], ...]), shape=(16 x N)
    """
    itvl = 16
    images_dir = os.path.join(dest_dir, 'images')
    positions_path = os.path.join(dest_dir, 'positions.npy')
    directions_path = os.path.join(dest_dir, 'directions.npy')
    
    if not os.path.exists(dest_dir): os.mkdir(dest_dir)
    if not os.path.exists(images_dir): os.mkdir(images_dir)
    
    num = 0
    num_exist = len(os.listdir(images_dir))
    positions_list = []
    directions_list = []
    print(f"Total {len(os.listdir(source_dir))} samples! ")
    for data in os.listdir(source_dir):
        d= os.path.join(source_dir, data)
        j_path = os.path.join(d, 'sln.json')

        f = open(j_path, 'r+')
        j = json.loads(f.read())
        img = cv2.imread(os.path.join(d, 'depth.png'))
        directions = [0]*itvl
        for angle in j['angle']:
            angle += 180
            angle %= 360
            d_index = int(angle/(360/itvl))
            directions[d_index] = 1
        keypoints = np.reshape(j['pullhold'], (2,2))
        new_img_path = os.path.join(images_dir, '%06d.png'%(num+num_exist))
        
        cv2.imwrite(new_img_path, img)
        positions_list.append(keypoints)
        directions_list.append(directions)
        num += 1
    
        if num % 100 == 0: 
            print(f"Transfer {num} samples! ") 
    np.save(positions_path, positions_list)
    np.save(directions_path, directions_list)

if __name__ == "__main__":
    src_dir = 'C:\\Users\\xinyi\\Documents\\XYBin_OnlyCalc\\bin\\exp' 
    # dest_dir = 'D:\\Dataset\\sepnet\\train'
    dest_dir = 'C:\\Users\\xinyi\\Documents\\Dataset\\sepnet'
    for _s in os.listdir(src_dir):
        if _s[0] == '_': continue
        _src_dir = os.path.join(src_dir, _s)
        _dest_dir = os.path.join(dest_dir, _s)
        print('--------', _src_dir, " => ", _dest_dir, '--------')
        gen_sep_data(_src_dir, _dest_dir)


