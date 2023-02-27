# Learning to Dexterously Pick or Separate Tangled-Prone Objects for Industrial Bin Picking

[Xinyi Zhang](http://xinyiz0931.github.io), Yukiyasu Domae, [Weiwei Wan](https://wanweiwei07.github.io/) and [Kensuke Harada](https://www.roboticmanipulation.org/members2/kensuke-harada/)      
Osaka University

[Webpage](http://xinyiz0931.github.io/tangle) / [arXiv](https://arxiv.org/abs/2302.08152) / [Video](https://youtu.be/O0y-Scp4wqY)  

## Overview  

<!-- ![teaser](image/harness_picking.jpg)  -->
Industrial bin picking for tangled-prone objects requires the robot to either pick up untangled objects or perform separation manipulation when the bin contains no isolated objects. The robot must be able to flexibly perform appropriate actions based on the current observation. It is challenging due to high occlusion in the clutter, elusive entanglement phenomena, and the need for skilled manipulation planning. In this paper, we propose an autonomous, effective and general approach for picking up tangled-prone objects for industrial bin picking. First, we learn PickNet - a network that maps the visual observation to pixel-wise possibilities of picking isolated objects or separating tangled objects and infers the corresponding grasp. Then, we propose two effective separation strategies: Dropping the entangled objects into a buffer bin to reduce the degree of entanglement; Pulling to separate the entangled objects in the buffer bin planned by PullNet - a network that predicts position and direction for pulling from visual input. To efficiently collect data for training PickNet and PullNet, we embrace the self-supervised learning paradigm using an algorithmic supervisor in a physics simulator. Real-world experiments show that our policy can dexterously pick up tangled-prone objects with success rates of 90%. We further demonstrate the generalization of our policy by picking a set of unseen objects.

This repository contains codes for training and inference using PickNet and PullNet. 

## Prerequisites

We're using a bin picking toolbox `bpbot: bin-picking-robot` containing some necessary functions such as grasp point detection. Please download and install this package. We've tested our code using Python 3.6, 3.7, 3.8. 

```
git clone https://github.com/xinyiz0931/bin-picking-robot.git bpbot
cd bpbot
pip insatll -r requirements.txt
pip install -e .
```

Then, install tensorflow for this repository. 

```
git clone https://github.com/xinyiz0931/tangle.git
```

We use PyTorch 1.8.1 and CUDA 11.1. 

## Training

Revise the training parameters in `cfg/config.yaml` and use

```
python tool/train.py
```

PickNet's dataset: 
```
    src_dir 
    ├── images
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── masks
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    └── labels.npy - shape=(N,), 0: picking, 1: separating
```
PullNet's dataset: 
```
    src_dir 
    ├── _images (before rotation)
    │   ├── 000000.png
    │   └── ...
    ├── _masks (before rotation)
    │   ├── 000000.png
    │   └── ...
    ├── _positions.npy (before rotation) - shape=(N,2)
    ├── _directions.npy (before rotation) - shape=(N,2)
    ├── images (after rotation, in use)
    │   ├── 000000.png
    │   └── ...
    ├── masks (after rotation, in use)
    │   ├── 000000.png
    │   └── ...
    └── positions.npy (after rotation, in use) - shape=(N,2)
```
