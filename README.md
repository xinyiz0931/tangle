# Tangle: PickNet + SepNet

[Xinyi Zhang](http://xinyiz0931.github.io), Yukiyasu Domae, [Weiwei Wan](https://wanweiwei07.github.io/) and [Kensuke Harada](https://www.roboticmanipulation.org/members2/kensuke-harada/)      
Osaka University

[arXiv](https://arxiv.org) / [Video](https://www.youtube.com)  

## Overview  

<!-- ![teaser](image/harness_picking.jpg)  -->

Robotic bin picking tasks for tangled-prone parts requires the robot to either lift the untangled objects or perform singulation manipulation when untangled objects do not exist. It is a challenging task due to the high-occluded scenes, elusive entanglement phenomena and skilled manipulation planning. The robot is also required to flexibly select suitable actions for the current observation. In this paper, we propose an autonomous, effective and generative approach for picking up tangled-prone objects in robotic bin picking. First, we learn a network to decide if the bin contains untangled objects and predict the grasp point for picking. If there are no such objects, we then learn a network to plan motions to separate them. Finally, we propose to transport the tangled objects from dense clutter to a transition area to reduce the degree of entanglement. Moreover, we embrace the self-supervised learning paradigm using a physics simulator. Results show a significant improvement in various bin picking tasks over baseline methods. 

Paper 
Thumbnal of paper

supplementary video 

Simulation video & more results

Real-world video & more results

SepNet performance video

Code usage
