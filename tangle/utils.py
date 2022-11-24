
import os
import json
import glob
import math
import random
import shutil
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

def test_cpp(string, number):
    return string*int(number)
    
def random_inds(sample_num, all_num):
    return random.sample(list(range(all_num)), sample_num)

def split_random_inds(num, val_ratio=0.2, test_ratio=None):
    inds = list(range(num))
    val_num = int(num*val_ratio)
    train_num = num - val_num
    if test_ratio == None:
        val_inds = random.sample(inds, val_num)
        train_inds = list(set(inds) - set(val_inds))
        return train_inds, val_inds
    else:
        test_num = int(num * test_ratio)
        rest_inds = random.sample(inds, (val_num+test_num))
        train_inds = list(set(inds) - set(rest_inds))
        val_inds = random.sample(rest_inds, (val_num))
        test_inds = list(set(rest_inds) - set(val_inds))
        return train_inds, val_inds, test_inds

def angle2keypoints(point, angle, width=10):
    """angle is in degree"""
    (x,y) = point
    rad = angle * math.pi / 180
    left_x = int(x + (width/2)*math.cos(rad))
    left_y = int(y - (width/2)*math.sin(rad))
    right_x = int(x - (width/2)*math.cos(rad))
    right_y = int(y + (width/2)*math.sin(rad))
    return left_x, left_y, right_x, right_y

def gauss_2d(img_w, img_h, sigma, loc, normalize_dist=False):
    """
    input: loc - numpy.array([u,v])
    output: G - torch.Size([img_w, img_h]), dtype=torch.double
    """
    import torch
    loc = np.array(loc)
    U, V = torch.from_numpy(loc)
    X, Y = torch.meshgrid([torch.arange(0., img_w), torch.arange(0., img_h)])
    X, Y = torch.transpose(X, 0, 1), torch.transpose(Y, 0, 1)
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    return G.double()
        
def gauss_2d_batch(img_w, img_h, sigma, locs, use_torch=True, use_cuda=False):
    """
    input: locs - numpy.array([[u1,v1],[u2,v2],...])
    output: 
        if use_torch: G - torch.Size([N, img_w, img_h]), dtype=torch.double
        else: G - array ([N, img_h, img_w], dtype=float)
    """
    locs = np.array(locs)
    if locs.shape == (2,): locs = np.asarray([locs])
    if use_torch: 
        import torch
        X,Y = torch.meshgrid([torch.arange(0., img_w), torch.arange(0., img_h)])
        X = torch.transpose(X, 0, 1)
        Y = torch.transpose(Y, 0, 1)

        U = torch.from_numpy(locs[:,0])
        V = torch.from_numpy(locs[:,1])
        U.unsqueeze_(1).unsqueeze_(2)
        V.unsqueeze_(1).unsqueeze_(2)

        G = torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
        return G.double().cuda() if use_cuda else G.double()

    else: 
        U = np.expand_dims(locs[:,0], axis=(-1,-2))
        V = np.expand_dims(locs[:,1], axis=(-1,-2))
        X,Y = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h))
        G = np.exp(-((X-U)**2+(Y-V)**2)/(2.0*sigma**2))
        return G

def draw_mask(img, msk, color='g'):
    if color == 'g': bgr = [0,255,0]
    elif color == 'b' or color == "blue": bgr = [255,102,51]
    elif color == 'pink': bgr = [144,89,244]

    if len(msk.shape) == 3:
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    
    cnt, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    drawn = cv2.drawContours(img,cnt,-1,bgr,2)  
    return drawn

def tensor_to_image(img_t, cmap=False):
    """
    input: img_t - torch.Size([img_w, img_h])
    """
    # img = img_t.cpu().numpy()
    img = img_t.detach().cpu().numpy()
    if len(img.shape) == 2: # (h,w)
        vis = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif img.shape[0] == 3 or (img.shape[0] == 1 and cmap is False): # (3,h,w) or (1,h,w)
        vis = np.moveaxis(img, 0, 2)
        vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif img.shape[0] == 1 and cmap is True: 
        vis = np.moveaxis(img, 0, 2) # (1,h,w)
        vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET) if cmap else vis
    elif img.shape[0] == 2 or img.shape > 3:
        img_c = []
        for i in range(img.shape[0]):
            _img = cv2.normalize(img[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _img = cv2.applyColorMap(_img, cv2.COLORMAP_JET) if cmap else _img
            img_c.append(_img)
        vis = cv2.hconcat(img_c)

    return vis 

def visualize_tensor(src_t, vis=True, cmap=False):
    """
    case 1: (H,W) or (1,H,W) or (H,W,1)
        - tensor to rgb image but channels are the same
    case 2: (3,H,W) or (H,W,3) or (1,3,H,W), or (1,W,H,3)
        - tensor to rgb image
    case 3: other cases 
    """
    src = src_t.detach().cpu().numpy()
    ret = src
    # case 1: (H,W), cmap? 
    if len(src.shape) == 2:
        if vis: 
            ret = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ret = cv2.applyColorMap(ret, cv2.COLORMAP_JET) if cmap else ret 

    # case 1: (1,H,W) or (H,W,1), vis/cmap=>(H,W)
    elif len(src.shape) == 3 and (src.shape[0] == 1 or src.shape[2] == 1):
        if vis: 
            src = np.squeeze(src)
            ret = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ret = cv2.applyColorMap(ret, cv2.COLORMAP_JET) if cmap else ret 
    # case 2: (3,H,W) or (H,W,3)  
    elif len(src.shape) == 3 and (src.shape[0] == 3 or src.shape[2] == 3):
        if vis: 
            if src.shape[0] == 3: src = np.moveaxis(src, 0, 2)
            ret = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.applyColorMap(ret, cv2.COLORMAP_JET) if cmap else vis

    # case 2: (1,3,H,W) or (1,H,W,3)
    elif len(src.shape) == 4 and src.shape[0] == 1 and (src.shape[1] == 3 or src.shape[3] == 3):
        if vis: 
            src = np.squeeze(src)
            if src.shape[0] == 3: src = np.moveaxis(src, 0, 2)
            ret = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.applyColorMap(ret, cv2.COLORMAP_JET) if cmap else vis
    return ret
    # elif src.shape[0] == 3 or (src.shape[0] == 1 and cmap is False): # (3,h,w) or (1,h,w)
    #     vis = np.moveaxis(src, 0, 2)
    #     vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # elif src.shape[0] == 1 and cmap is True: 
    #     vis = np.moveaxis(src, 0, 2) # (1,h,w)
    #     vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET) if cmap else vis
    # elif src.shape[0] == 2 or src.shape[0] > 3:
    #     src_c = []
    #     for i in range(src.shape[0]):
    #         _src = cv2.normalize(src[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #         _src = cv2.applyColorMap(_src, cv2.COLORMAP_JET) if cmap else _src
    #         src_c.append(_src)
    #     vis = cv2.hconcat(src_c)
    # else:
    #     vis = src 
    # return ret 
    #cv2.imshow("windows", vis)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def bilinear_init(in_channels, out_channels, kernel_size):
    import torch
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size,:kernel_size]
    filt = (1-abs(og[0]-center)/factor) * (1-abs(og[1]-center)/factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

def cross_entropy2d(input, target, weight=None, size_average=True):
    from distutils.version import LooseVersion
    import torch
    import torch.nn.functional as F
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

# def vector2direction(drag_v):
#     """
#     input: drag_v - 2d normalized vector
#     output: direction (degree) of image counterclockwise-rotation which makes drag_v equals [1,0]
#     """
#     from bpbot.utils import calc_2vectors_angle
#     degree_x = calc_2vectors_angle(drag_v, [1,0])
#     degree_y = calc_2vectors_angle(drag_v, [0,1])
#     if degree_y <= 90: rot_degree = degree_x
#     else: rot_degree = -degree_x
#     return rot_degree

def vector2angle(v):
    """
    input: drag_v - 2d normalized vector
    output: direction (degree) of image counterclockwise-rotation which makes v equals [1,0]
    """
    from bpbot.utils import calc_2vectors_angle
    degree_x = calc_2vectors_angle(v, [1,0])
    degree_y = calc_2vectors_angle(v, [0,1])
    if degree_y <= 90: rot_degree = degree_x
    # from the old function... 
    else: rot_degree = 360 - degree_x
    # from often used vector2direction()
    # else: rot_degree = -degree_x
    return rot_degree

def angle2vector(r, point_to='right'):
    from bpbot.utils import rotate_point
    if point_to == 'right':
        v = rotate_point([1,0], r)
    elif point_to == 'left':
        v = rotate_point([-1, 0], r)
    return np.array(v) / np.linalg.norm(np.array(v))

# def direction2vector(rot_degree):
#     """
#     input: rotate_degree - angle of image counterclockwise-rotation which has vector [1,0]
#     output: vector of dragging on pre-rotation image
#     rotate the [1,0] clockwise for rot_degree
#     """
#     from bpbot.utils import rotate_point
#     # [1,0] -> drag_v_norm
#     drag_v = rotate_point([1,0], rot_degree)
#     drag_v = np.array(drag_v) / np.linalg.norm(np.array(drag_v)) # 2d norm
#     return drag_v

def draw_vector(src, p, v, arrow_len=None, arrow_thickness=2, color=(0,255,0)):
    """
    drag_v: 2d normalizaed vector
    """
    img = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB) if len(src.shape) == 2 else src
    h, w, _ = img.shape

    if arrow_len == None: arrow_len = int(h/10)

    stop_p = [int(p[0]+v[0]*arrow_len), int(p[1]+v[1]*arrow_len)]
    color = (color[2], color[1], color[0]) # rgb --> bgr
    # find drawable region
    if stop_p[0] > w: stop_p[0] = int(p[0]+v[0]*(w-p[0]-5))
    if stop_p[1] > h: stop_p[1] = int(p[1]+v[1]*(h-p[1]-5))
    if stop_p[0] < 0: stop_p[0] = int(p[0]+v[0]*(p[0]+5))
    if stop_p[1] < 0: stop_p[1] = int(p[1]+v[1]*(p[1]+5))
    drawn = cv2.arrowedLine(img, p, stop_p, color, arrow_thickness)
    if len(src.shape) == 2:
        return cv2.cvtColor(drawn, cv2.COLOR_RGB2GRAY)
    return drawn

# def draw_vector(img, start_p, v, arrow_len=None, arrow_thickness=2, color=(0,255,255)):
#     """
#     drag_v: 2d normalizaed vector
#     color: (r,g,b)
#     """
#     h, w, _ = img.shape
#     if arrow_len == None: arrow_len = int(h/10)

#     start_p = [int(start_p[0]), int(start_p[1])]
#     stop_p = [int(start_p[0]+v[0]*arrow_len), int(start_p[1]+v[1]*arrow_len)]
#     color_bgr = (color[2], color[1], color[0]) # rgb --> bgr
#     # find drawable region
#     if stop_p[0] > w: stop_p[0] = int(start_p[0]+v[0]*(w-start_p[0]-5))
#     if stop_p[1] > h: stop_p[1] = int(start_p[1]+v[1]*(h-start_p[1]-5))
#     if stop_p[0] < 0: stop_p[0] = int(start_p[0]+v[0]*(start_p[0]+5))
#     if stop_p[1] < 0: stop_p[1] = int(start_p[1]+v[1]*(start_p[1]+5))
#     drawn = cv2.arrowedLine(img, start_p,stop_p, color_bgr, arrow_thickness)
#     return drawn

def draw_vectors_bundle(img, start_p, scores=None, scores_thre=0.4, itvl=16):
    if scores is None: 
        scores = list(range(itvl))
    else: 
        itvl = len(scores)
    # top_five = sorted(range(len(scores)), key=lambda i: scores[i])[-5:-1]
    top_r_index = sorted(range(len(scores)), key=lambda i: scores[i])[-1]
    for r, s in enumerate(scores):
        if s > scores_thre: # success
            draw_vector(img, start_p, angle2vector(r*360/itvl), arrow_thickness=1,color=(0,255,0))
            # print("success: ", r*360/itvl, scores.index(s))
        else: # failure
            draw_vector(img, start_p, angle2vector(r*360/itvl), arrow_thickness=1,color=(255,0,0))
            # print("fail: ",r*360/itvl, scores.index(s))
    top_r = top_r_index*360/itvl
    top_v = angle2vector(top_r)

    if len(scores) == 2 and (np.unique(scores) == [0,1]).all(): 
        return img
    
    if scores[top_r_index] > scores_thre:
        draw_vector(img, start_p, top_v, arrow_thickness=2,color=(0,255,0))
    else:
        draw_vector(img, start_p, top_v, arrow_thickness=2,color=(255,255,0))
    return img
    cv2.imshow("windows", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sample_directions(itvl=16):
    directions = []
    for r in range(itvl):
        directions.append(angle2vector(r*360/itvl))
    return directions
# def transfer_data(src_dir, dest_dir, option, success):
#     """
#     condition: option="drag"/"pick", success=True/False
#     """
#     print(f"Start transfer {option} data with only success == {success}")
#     num = 0
#     if not os.path.exists(dest_dir): os.mkdir(dest_dir)
#     for work in os.listdir(src_dir):
#         work_dir = os.path.join(src_dir, work)
        
#         # for data in os.listdir(work_dir):
#         for data in (os.listdir(work_dir))[::-1]:
#             d = os.path.join(work_dir, data)
#             f = open(os.path.join(d, "info.json"), "r")
#             j = json.loads(f.read())
            
#             # transfer drag data with all
#             if option == "drag" and "drag" in j:
#                 if not os.path.exists(os.path.join(dest_dir, data)): 
#                     shutil.copytree(d, os.path.join(dest_dir, data))
#                     num += 1
#             # transfer drag data with only success
#             elif option == "drag" and "drag" in j and j["success"]:
#                 if not os.path.exists(os.path.join(dest_dir, data)): 
#                     shutil.copytree(d, os.path.join(dest_dir, data))
#                     num += 1

#                 num += 1
#             # transfer pick data with all
#             elif option == "pick" and not "drag" in j:
#                 if not os.path.exists(os.path.join(dest_dir, data)): 
#                     shutil.copytree(d, os.path.join(dest_dir, data))
#                     num += 1

#                 num += 1
#             # transfer pick data with only success
#             elif option == "pick" and not "drag" in j and j["success"]:
#                 if not os.path.exists(os.path.join(dest_dir, data)): 
#                     shutil.copytree(d, os.path.join(dest_dir, data))
#                     num += 1

#                 num += 1
            
#             # print 
#             if num % 100 == 0 and num != 0:
#                 print(f"trasferred {(num)} data ..")

# def reform_pick_data(src_folder):
#     """
#     0. rename: 20200101051100 --> 000001
#     1. update json file
#         ["id"]
#     2. visualization
#         vis.pnt
#         delete grasp.png if exists
#     """
#     from bpbot.grasping import Gripper
#     gripper = Gripper(finger_w=12, finger_h=30, open_w=40, gripper_size=500)

#     num = 0
#     for data in os.listdir(src_folder):
#         d = os.path.join(src_folder, data)
#         new_d = os.path.join(src_folder, "%06d"%num)
#         f = open(os.path.join(d, "info.json"), "r+")
#         j = json.loads(f.read())
#         img = cv2.imread(os.path.join(d, "depth.png"))
#         if not 'drag' in j:
#             # 1. update json file
#             f.seek(0)
#             json.dump(j,f, indent=4)
#             f.truncate()
#             # 2. visualization
#             grasp_p = [j["pick"]["point"][0], j["pick"]["point"][1]]
#             grasps = [[None, grasp_p[0], grasp_p[1],None,j["pick"]["angle"]*math.pi/180]]
#             drawn = gripper.draw_grasp(grasps, img.copy())
#             cv2.imwrite(os.path.join(d, "vis.png"), drawn)
            
#             # 0. rename
#             if os.path.exists(os.path.join(d, "grasp.png")):
#                 os.remove(os.path.join(d, "grasp.png"))
#             os.rename(d, new_d)
#             num += 1

# def reform_drag_data(src_folder):
#     """
#     0. rename: 20200101051100 ==> 000001
#     1. update json file: 
#         ["id"]
#         ["pick"]["angle_rot"]
#         ["pick"]["point_rot"]
#         ["drag"]["vector2d"]
#         ["drag"]["angle"]
#     2. rotate image
#         rot.png, rot_mask.png, rot_mask_f.png
#     3. visualization
#         vis.png, rot_vis.png
#         delete grasp.png if exists
#     """
#     from bpbot.grasping import Gripper
#     gripper = Gripper(finger_w=12, finger_h=30, open_w=40, gripper_size=500)
#     num = 0
#     for data in os.listdir(src_folder):
#         # mkdir dest
#         d = os.path.join(src_folder, data)
#         new_d = os.path.join(src_folder, "%06d"%num)
#         f = open(os.path.join(d, "info.json"), "r+")
#         j = json.loads(f.read())
#         img = cv2.imread(os.path.join(d, "depth.png"))
#         msk = cv2.imread(os.path.join(d, "mask_target.png"))
#         msk_f = cv2.imread(os.path.join(d, "mask_others.png"))
#         if 'drag' in j and j["success"]:
#             sim_v = j["drag"]["vector"] # 3d
#             drag_v = [-round(sim_v[2],3), round(sim_v[0],3)] # 2d
#             drag_v_norm = np.array(drag_v) / np.linalg.norm(np.array(drag_v)) # 2d norm
            
#             grasp_p = [j["pick"]["point"][0], j["pick"]["point"][1]]
#             # start rotating
#             #degree_x = calc_2vectors_angle(drag_v, [1,0]) 
#             #degree_y = calc_2vectors_angle(drag_v, [0,1])
#             #if degree_y <= 90: rot_degree = degree_x
#             #else: rot_degree = -degree_x
#             rot_degree = vector2direction(drag_v_norm)
#             grasp_p_rot = rotate_pixel(img.shape[0],img.shape[1], grasp_p, rot_degree, cropped=False)
            
#             # 0. rename
#             # 1. update json file
#             j["pick"]["angle_rot"] = (rot_degree+j["pick"]["angle"])%360
#             j["pick"]["point_rot"] = [grasp_p_rot[0], grasp_p_rot[1]]
#             j["drag"]["vector2d"] =[drag_v_norm[0],drag_v_norm[1]] 
#             j["drag"]["angle"]=rot_degree

#             f.seek(0)
#             json.dump(j,f, indent=4)
#             f.truncate()
#             # 2. save the rotated image
#             rot = rotate_img(img, rot_degree, cropped=False)
#             cv2.imwrite(os.path.join(d, "rot.png"), rot)
#             rot_msk = rotate_img(msk, rot_degree, cropped=False)
#             cv2.imwrite(os.path.join(d, "rot_msk.png"), rot_msk)
#             rot_msk_f = rotate_img(msk_f, rot_degree, cropped=False)
#             cv2.imwrite(os.path.join(d, "rot_msk_f.png"), rot_msk_f)
#             # 3. draw and save the vis result
#             grasps = [[None, grasp_p[0], grasp_p[1],None,j["pick"]["angle"]*math.pi/180]]
#             drawn = gripper.draw_grasp(grasps, img.copy())
#             drawn = draw_drag_vector(drawn, grasp_p, drag_v_norm, 200)
#             grasps_rot = [[None, grasp_p_rot[0], grasp_p_rot[1], None, j["pick"]["angle_rot"]*math.pi/180]]

#             drawn_rot = gripper.draw_grasp(grasps_rot, rot.copy())
#             drawn_rot = draw_drag_vector(drawn_rot, grasp_p_rot, [1,0], 200)

#             cv2.imwrite(os.path.join(d, "vis.png"), drawn)
#             cv2.imwrite(os.path.join(d, "rot_vis.png"), drawn_rot)
            
#             # 0. rename
#             if os.path.exists(os.path.join(d, "grasp.png")):
#                 os.remove(os.path.join(d, "grasp.png"))
#             os.rename(d, new_d)
#             num += 1