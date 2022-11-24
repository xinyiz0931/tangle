
import os
import json
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from tangle.utils import *
from bpbot.utils import *

seq = iaa.Sequential([
    iaa.Affine(
        scale=(0.9,1.1),
        shear=(-10,10),
        rotate=(-180,180)
    ),
    # iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.1)),
    # iaa.Sometimes(0.5, iaa.GammaContrast((0.5, 2.0))),
    # iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=1, sigma=1))
    ])
# seq = iaa.Sequential([])
seq_noise_only = iaa.Sequential([
    iaa.Affine(
        scale=(0.9,1.1),
        shear=(-10,10),
    ),
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.1)),
    iaa.Sometimes(0.5, iaa.GammaContrast((0.5, 2.0))),
    iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=1, sigma=1))
    ])

def transfer_and_refine(solution_dir, source_dir):
    from bpbot.grasping import Gripper
    itvl = 16
    gripper = Gripper(finger_h=20, finger_w=13, open_w=38)
    poses_sln = []
    poses_src = []
    for i in os.listdir(solution_dir):
        a = np.loadtxt(os.path.join(solution_dir, i, "pose.txt"))
        poses_sln.append(a[0])

    for i in os.listdir(source_dir):
        b = np.loadtxt(os.path.join(source_dir, i, "pose.txt"))
        poses_src.append(b[0])
    dict = {}
    for i, pi in eNerate(poses_sln):
        for j, pj in eNerate(poses_src):
            if np.all(pi==pj)==True: 
                scene_no = os.listdir(source_dir)[j]
                sln_no = os.listdir(solution_dir)[i]
                if not scene_no in dict: dict[scene_no]=[]
                dict[scene_no].append(sln_no)
                # print("(",i,")",os.listdir(solution_dir)[i], " ==> ", os.listdir(source_dir)[j])
                # print("(",i,")", sln_no, " ==> ", scene_no)
                # json_dir = os.path.join(solution_dir, os.listdir(solution_dir)[i], "info.json")
                # f = open(json_dir, "r+")
                # json_file = json.loads(f.read())
                
                # # update json file
                # json_file["index"] = int(os.listdir(source_dir)[j])
                # f.seek(0)
                # json.dump(json_file, f, indent=4)
                # f.truncate()
    from tangle.utils import vector2direction

    for s in dict:
        print(f"========== scene {s} ==========")
        subdict = {}
        rot_list = []
        pos_list = []
        
        for d in dict[s]: 
            json_dir = os.path.join(solution_dir, d, "info.json")
            f = open(json_dir, "r")
            json_file = json.loads(f.read())
            sim_v = json_file["drag"]["vector"] # 3d
            drag_v = [-round(sim_v[2],3), round(sim_v[0],3)] # 2d
            drag_v_norm = np.array(drag_v) / np.linalg.norm(np.array(drag_v)) # 2d norm
            rot_degree = vector2direction(drag_v_norm)
            # revised: all rotation degrees are positive
            if rot_degree < 0: rot_degree = 360+rot_degree
            pull_p = [json_file["pick"]["point"][0], json_file["pick"]["point"][1]]
            pull_a = json_file["pick"]["angle"]
            hold_p = [json_file["assist"]["point"][0], json_file["assist"]["point"][1]]
            hold_a = json_file["assist"]["angle"]
            # refine the grasp pose
            pull_id = json_file["pick"]["id"]
            hold_id = json_file["assist"]["id"]
            _pm = cv2.imread(os.path.join(solution_dir, d, f"mask_{pull_id}.png"))
            _hm = cv2.imread(os.path.join(solution_dir, d, f"mask_{hold_id}.png"))
            msk = cv2.imread(os.path.join(solution_dir, d, "mask_target.png"), 0)
            
            img_h, img_w = msk.shape
            
            _pm = cv2.imread(os.path.join(solution_dir, d, f"mask_{pull_id}.png"))
            _hm = cv2.imread(os.path.join(solution_dir, d, f"mask_{hold_id}.png"))
            
            # hold mask / crossing region
            hold_mask = cv2.cvtColor(cv2.bitwise_and(_pm, _hm), cv2.COLOR_RGB2GRAY)
            hold_mask = adjust_array_range(hold_mask, if_img=True)

            rect = gripper.get_hand_model('close',img_h,img_w,open_width=50,
                                            x=pull_p[0],y=pull_p[1],radian=pull_a*math.pi/180)
            # pull mask / grasp rect
            grasp_rect = cv2.bitwise_and(msk, rect)

            crossing_contours, _ = cv2.findContours(hold_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            min_dist = 9999
            hold_p_mask_center = [0,0]
            for cnt in crossing_contours:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    dist = calc_2points_distance([cx, cy], hold_p)
                    if dist <= min_dist: 
                        min_dist = dist
                        # hold_p_ = hold_p
                        hold_p_mask_center = [cx, cy]
                        hold_p_mask = np.zeros_like(_pm) # 3-channel
                        cv2.drawContours(hold_p_mask, [cnt], -1, color=(255,255,255), thickness=-1)

            grasp_contours, _ = cv2.findContours(grasp_rect, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(grasp_contours) == 0: print(d)
            M = cv2.moments(grasp_contours[0])
            
            if M['m00'] != 0: 
                grasp_rect_center = [int(M['m10']/M['m00']), int(M['m01']/M['m00'])]

            
            # end of refinement
            rot_list.append(np.round(rot_degree, 1))

            pos_list.append([grasp_rect_center[0], grasp_rect_center[1], 
                        hold_p_mask_center[0],hold_p_mask_center[1]])

        # print(dict[s])
        # print(pos_list, rot_list)
        N = 0
        grasps, directions, solutions = [], [], []
        for p in pos_list:
            if not p in grasps:
                grasps.append(p)
        
        for g in grasps:
            _rot = []
            _sln = []
            for index, p in eNerate(pos_list):
                if g == p: 
                    if not rot_list[index] in _rot:
                        _rot.append(rot_list[index])
                        _sln.append(dict[s][index])
            directions.append(_rot)
            solutions.append(_sln)
        subdict["pullhold"] = grasps
        subdict["angle"] = directions
        subdict["solutions"] = solutions

        new_jf_path = os.path.join(source_dir, s, "info.json")
        # print(new_jf_path)
        # print(subdict)

        # with open(new_jf_path, 'w', encoding='utf-8') as jf:
        #     json.dump(subdict, jf, ensure_ascii=False, indent=4) 
        # jf.close()

def vis_solution(source_dir, dest_dir):
    N = 0
    if not os.path.exists(dest_dir): os.mkdir(dest_dir)
    itvl = 16
    for s in os.listdir(source_dir):
        img = cv2.imread(os.path.join(source_dir, s, "depth.png"))
        json_dir = os.path.join(source_dir, s, "info.json")
        f = open(json_dir, "r")
        json_file = json.loads(f.read())
        if "pullhold" in json_file:
            grasps = json_file["pullhold"]
            directions = json_file["angle"]

            for i, g in eNerate(grasps):
                drawn = img.copy()
                
                for j in range(itvl):
                    r = j*(360/itvl)
                    if r in directions[i]:
                        draw_vector(drawn, (g[0], g[1]), direction2vector(r), 40, arrow_thinkness=2,color=(0,255,0))
                    else:
                        draw_vector(drawn, (g[0], g[1]), direction2vector(r), 40, arrow_thinkness=1,color=(0,0,255))
                drawn = cv2.circle(drawn, (g[0],g[1]),7,(0,255,0),-1)
                # drawn = cv2.circle(drawn, (g[2],g[3]),7,(0,255,0),-1)
                save_path = os.path.join(dest_dir, "%06d.png"%N)
                cv2.imwrite(save_path, drawn)
                N += 1
                #cv2.imshow(s, drawn)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

def simplify_source_data(source_dir, dest_dir):
    for data in os.listdir(source_dir):
        
        d = os.path.join(source_dir, data)
        json_path = os.path.join(d, "info.json")
        with open(json_path, 'r+') as f:
            j = json.loads(f.read())
        
        x = j["pick"]["point"][0]
        y = j["pick"]["point"][1]
        theta = j["pick"]["angle"]
        if "drag" in j: label = 1
        else: label = 0

        info = f"{label}_{x}_{y}_{theta}"
        img_path = os.path.join(d, "depth.png")
        msk_path = os.path.join(d, "mask_target.png")
        gsp_path = os.path.join(d, "grasp.png")

        new_img_path = os.path.join(dest_dir, info+'.png')
        new_msk_path = os.path.join(dest_dir, info+'_m.png')
        new_gsp_path = os.path.join(dest_dir, info+'_g.png')

        shutil.copyfile(img_path, new_img_path)
        shutil.copyfile(msk_path, new_msk_path)
        # shutil.copyfile(gsp_path, new_gsp_path)
def gen_pickdata(source_dir, dest_dir, aug_multiplier=1):


    images_dir = os.path.join(dest_dir, "images")
    masks_dir = os.path.join(dest_dir, "masks")
    grasps_dir = os.path.join(dest_dir, "grasps")
    labels_path = os.path.join(dest_dir, "labels.npy")

    for d in [dest_dir, images_dir, masks_dir, grasps_dir]:
        if not os.path.exists(d): os.mkdir(d)
    N = 0
    N_exist = len(os.listdir(images_dir))
    if os.path.exists(labels_path): 
        labels = np.load(labels_path).tolist()
    else: 
        labels = []
    from bpbot.grasping import Gripper
    gripper = Gripper(finger_h=20, finger_w=13, open_w=38)

    print(f"[!] Already exists {N_exist} samples! ") 
    print(f"[*] Total {len(os.listdir(source_dir))} samples! ")
    for data in os.listdir(source_dir):
        d = os.path.join(source_dir, data)
        json_path = os.path.join(d, "info.json")
        with open(json_path, "r+") as f:
            j = json.loads(f.read())
        p = j["pick"]["point"]
        theta = j["pick"]["angle"]
        img = cv2.imread(os.path.join(d, "depth.png"))
        img_h, img_w, _ = img.shape
        msk = cv2.imread(os.path.join(d, "mask_target.png"), 0)
        rect = gripper.get_hand_model('close',img_w,img_h,open_w=40,x=p[0],y=p[1],theta=theta)
        gsp = cv2.bitwise_and(msk, rect)

        if "drag" in j:
            # negative samples: augment 10x
            
            msk = cv2.normalize(msk, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
            gsp = cv2.normalize(gsp, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
            heatmap = np.stack([msk, gsp], 2)
            heatmap = HeatmapsOnImage(heatmap, shape=img.shape)
            
            for _ in range(aug_multiplier):
                # images_aug = seq(images=images)
                # image_aug, segmaps_aug = seq(image=images, segmentation_maps=segmaps)
                img_, masks_ = seq(image=img, heatmaps=heatmap)
                msk_ = masks_.get_arr()[:,:,0]*255 
                gsp_ = masks_.get_arr()[:,:,1]*255
                msk_ = cv2.normalize(msk_, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                gsp_ = cv2.normalize(gsp_, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                 
                new_img_path = os.path.join(images_dir, "%06d.png" % (N+N_exist))
                new_msk_path = os.path.join(masks_dir, "%06d.png" % (N+N_exist))
                new_gsp_path = os.path.join(grasps_dir, "%06d.png" % (N+N_exist))

                # cv2.imwrite(new_img_path, img_)
                # cv2.imwrite(new_msk_path, msk_)
                # cv2.imwrite(new_gsp_path, gsp_)
                N += 1 
                labels.append([1, p[0],p[1], theta])
        else:
            # positive samples: randomly add noise
            img = seq_noise_only(image=img)
            new_img_path = os.path.join(images_dir, "%06d.png" % (N+N_exist))
            new_msk_path = os.path.join(masks_dir, "%06d.png" % (N+N_exist))
            new_gsp_path = os.path.join(grasps_dir, "%06d.png" % (N+N_exist))

            # cv2.imwrite(new_img_path, img)
            # cv2.imwrite(new_msk_path, msk)
            # cv2.imwrite(new_gsp_path, gsp)
            N += 1
            labels.append([0, p[0],p[1], theta])
        # if N > 5: break
        print("Transfer data: %6d" % N, end='')
        print('\r', end='')
    # np.save(labels_path, labels)
    
def gen_simple_pickdata(source_dir, dest_dir, type="pick", aug_multiplier=1):
    images_dir = os.path.join(dest_dir, "images")
    masks_dir = os.path.join(dest_dir, "masks")
    labels_path = os.path.join(dest_dir, "labels.npy")

    for d in [images_dir, masks_dir]:
        if not os.path.exists(d): os.mkdir(d)
    N = 0
    N_exist = len(os.listdir(images_dir))
    if os.path.exists(labels_path):
        labels = np.load(labels_path).tolist()
    else:
        labels = []

    print(f"[!] Already exists {N_exist} samples! ") 
    print(f"[*] Total {len(os.listdir(source_dir))} samples! ")

    if type == "pick":
        for data in os.listdir(source_dir):
            d = os.path.join(source_dir, data)
            json_path = os.path.join(d, "info.json")
            with open(json_path, "r+") as f:
                j = json.loads(f.read())
            p = j["pick"]["point"]
            theta = j["pick"]["angle"]
            img = cv2.imread(os.path.join(d, "depth.png"))
            # img_h, img_w, _ = img.shape
            msk = cv2.imread(os.path.join(d, "mask_target.png"), 0)

            if "drag" in j:
                # negative samples: augment 10x
                msk = cv2.normalize(msk, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
                heatmap = HeatmapsOnImage(msk, shape=img.shape)
                for _ in range(aug_multiplier):
                    img_, masks_ = seq(image=img, heatmaps=heatmap)
                    msk_ = cv2.normalize(masks_.get_arr(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    new_img_path = os.path.join(images_dir, "%06d.png" % (N+N_exist))
                    new_msk_path = os.path.join(masks_dir, "%06d.png" % (N+N_exist))

                    cv2.imwrite(new_img_path, img_)
                    cv2.imwrite(new_msk_path, msk_)
                    N += 1 
                    labels.append(0)
            else:
                # positive samples: randomly add noise
                img = seq_noise_only(image=img)
                new_img_path = os.path.join(images_dir, "%06d.png" % (N+N_exist))
                new_msk_path = os.path.join(masks_dir, "%06d.png" % (N+N_exist))

                cv2.imwrite(new_img_path, img)
                cv2.imwrite(new_msk_path, msk)
                N += 1
                labels.append(0)
            print("Transfer data: %6d" % N, end='')
            print('\r', end='')
    
    elif type == "sep":
        for data in os.listdir(source_dir):
            d = os.path.join(source_dir, data)
            img = cv2.imread(os.path.join(d, os.listdir(d)[0]))
            for msk_path in [x for x in os.listdir(d) if "mask" in x and not "target" in x]:
                msk = cv2.imread(os.path.join(d, msk_path), 0)
                
                msk = cv2.normalize(msk, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
                heatmap = HeatmapsOnImage(msk, shape=img.shape)
                for _ in range(aug_multiplier):
                    img_, masks_ = seq(image=img, heatmaps=heatmap)
                    msk_ = cv2.normalize(masks_.get_arr(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    new_img_path = os.path.join(images_dir, "%06d.png" % (N+N_exist))
                    new_msk_path = os.path.join(masks_dir, "%06d.png" % (N+N_exist))

                    cv2.imwrite(new_img_path, img_)
                    cv2.imwrite(new_msk_path, msk_)
                    labels.append(1)
                    N += 1 
                
                print("Transfer data: %6d" % N, end='')
                print('\r', end='')

    np.save(labels_path, labels)

def gen_sepdata_from_pe(source_dir, dest_dir, itvl=16):
    """
    Generate dastaset from only calculated data, using directory/.json
    ├ dest_dir
    ├── images
    │   ├── 000000.png
    │   └── ...
    ├── direction.npz - np.array([[pull vecotor x,y], [...], ...]), shape = (16 x 2)
    ├── positions.npz - np.array([[pull_x, pull_y, hold_x, hold_y], [...], ...]), shape = (16 x N)
    └── labels.npz - np.array([[1,0,0,0,...], [...], ...]), shape=(itvl x N)
    Source_dir: contains `sln.json`
    Directions are at image coordinates, default as starting from pointing right and in counter-clockwise
    """
    images_dir = os.path.join(dest_dir, "images")
    direction_path = os.path.join(dest_dir, "direction.npy")
    positions_path = os.path.join(dest_dir, "positions.npy")
    labels_path = os.path.join(dest_dir, "labels.npy")
    
    if not os.path.exists(dest_dir): os.mkdir(dest_dir)
    if not os.path.exists(images_dir): os.mkdir(images_dir)
    
    N = 0
    N_exist = len(os.listdir(images_dir))
    
    if os.path.exists(positions_path):
        positions_list = np.load(positions_path).tolist()
    else: 
        positions_list = []
    if os.path.exists(labels_path):
        labels_list = np.load(labels_path).tolist()
    else:
        labels_list = []
    direction_list = []
    for i in range(itvl): 
        direction_list.append(angle2vector(i*360/itvl))
    
    print(f"[!] Already exists {len(positions_list)} samples! ") 
    print(f"[*] Total {len(os.listdir(source_dir))} samples! ")
    for data in os.listdir(source_dir):
        d = os.path.join(source_dir, data)
        j_path = os.path.join(d, "sln.json")
        if not os.path.exists(j_path): 
            continue

        fp = open(j_path, 'r+')
        json_file = json.loads(fp.read())
        if not json_file: continue
        img = cv2.imread(os.path.join(d, "depth.png"))
        l = [0] * itvl
        if "pullhold" in json_file:
            grasps = np.array(json_file["pullhold"], dtype=int)
            degrees = json_file["angle"]
            if len(grasps.shape) == 1:
                g = np.reshape(grasps, (2,2))
                for j in range(itvl):
                    r = j*(360/itvl)
                    if r in degrees:
                        l[j] = 1
            else: 
                for i, g in eNerate(grasps):
                    g = np.reshape(g, (2,2))
                    for j in range(itvl):
                        r = j*(360/itvl)
                        if r in degrees[i]:
                            l[j] = 1
        
        elif "pull" in json_file and "hold" in json_file:
            degrees = json_file["angle"]
            g = np.array([json_file["pull"], json_file["hold"]])
            for k in range(itvl):
                r = k*(360/itvl)
                if r in degrees: 
                    l[k] = 1

        i_list, g_list, l_list = augment_data(image=img, grasp=g, label=l, aug_rot_itvl=1, aug_multiplier=1) 
        # i_list, g_list, l_list = augment_data(image=img, grasp=g, label=l, aug_rot_itvl=4, aug_multiplier=3) 
        for i_, g_, l_ in zip(i_list, g_list, l_list):
            new_img_path = os.path.join(images_dir, "%06d.png" % (N+N_exist))
            positions_list.append(g_)
            labels_list.append(l_)
            cv2.imwrite(new_img_path, i_)
            N += 1
            print('[*] Generating data: %d' % (N), end='')
            print('\r', end='') 
        #     if N >= 20: break
        # if N >= 20: break

    print(f"[*] Finish generating {N} samples! ")
            # if N % 100 == 0: print(f"Transferred {N} samples! ")
    np.save(positions_path,positions_list)
    np.save(labels_path, labels_list)
    np.save(direction_path, direction_list) 

def gen_simple_sepdata(source_dir, dest_dir, aug_multimplier=1):
    _images_dir = os.path.join(dest_dir, "_images")
    images_dir = os.path.join(dest_dir, "images")
    _masks_dir = os.path.join(dest_dir, "_masks")
    masks_dir = os.path.join(dest_dir, "masks")

    _directions_path = os.path.join(dest_dir, "_directions.npy")
    _positions_path = os.path.join(dest_dir, "_positions.npy")
    positions_path = os.path.join(dest_dir, "positions.npy")
    
    for d in [dest_dir, images_dir, masks_dir, _images_dir, _masks_dir]:
        if not os.path.exists(d): os.mkdir(d)
    
    N = 0 
    N_exist = len(os.listdir(images_dir))
    positions_list, _positions_list, _directions_list = [], [], []
    if os.path.exists(positions_path):
        positions_list = np.load(positions_path).tolist()
    if os.path.exists(_directions_path):
        _directions_list = np.load(_directions_path).tolist()
    if os.path.exists(_positions_path):
        _positions_list = np.load(_positions_path).tolist()

    print(f"[!] Already exists {N_exist} samples! ")
    print(f"[*] Total {len(os.listdir(source_dir)) * aug_multimplier} samples! ")
    for data in os.listdir(source_dir):
        d = os.path.join(source_dir, data)
        
        # ['281_165_-1.000000_0.000000.png', 'color.png', 'grasp.png', 'mask_0.png', 'mask_1.png', 'pose.txt']
        img_path = os.path.join(d, os.listdir(d)[0])
        # img_path = os.path.join(d, "grasp.png")
        msk_path = os.path.join(d, "mask_target.png")
        img = cv2.imread(img_path)
        msk = cv2.imread(msk_path)
        x,y,vx,vy = os.listdir(d)[0].split("_")
        x = int(x)
        y = int(y)
        vx = float(vx)
        vy = float(vy[:-4])
        va = vector2angle([vx,vy])
        rot_img, rot_p = rotate_img_kpt(img, [[x,y]], va)
        rot_msk = rotate_img(msk, va) 
        rot_p = rot_p[0].astype(int)

        # augmentatiaon
        sgmap = SegmentationMapsOnImage(rot_msk, shape=rot_img.shape)
        kpt = KeypointsOnImage([Keypoint(x=rot_p[0],y=rot_p[1])], shape=rot_img.shape)
        for _ in range(aug_multimplier):
            img_aug, kpt_aug, sgmap_aug = seq_noise_only(image=rot_img, keypoints=kpt, segmentation_maps=sgmap)
            pos_aug = np.array([kpt_aug[0].x, kpt_aug[0].y], dtype=int)
            msk_aug = cv2.normalize(sgmap_aug.get_arr(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) 

            # drawn =draw_mask(img_aug, msk_aug, color='blue')  
            # cv2.circle(img_aug, pos_aug, 7, (0,255,0), -1)
            # cv2.imshow("", drawn)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            _new_img_path = os.path.join(_images_dir, "%06d.png" % (N+N_exist))
            _new_msk_path = os.path.join(_masks_dir, "%06d.png" % (N+N_exist))
            new_img_path = os.path.join(images_dir, "%06d.png" % (N+N_exist))
            new_msk_path = os.path.join(masks_dir, "%06d.png" % (N+N_exist))
            _positions_list.append([x,y])
            _directions_list.append([vx,vy])
            positions_list.append(pos_aug)
            cv2.imwrite(_new_img_path, img)
            cv2.imwrite(_new_msk_path, msk)
            cv2.imwrite(new_img_path, img_aug)
            cv2.imwrite(new_msk_path, msk_aug)

            N += 1

        print('[*] Generating data: %d' % (N), end='')
        print('\r', end='') 

    print(f"[*] Finish generating {N} samples! ")
    np.save(positions_path, positions_list)
    np.save(_positions_path, _positions_list)
    np.save(_directions_path, _directions_list) 
    
def augment_simple_data(image, position, aug_multiplier=4):
    images_aug = []
    positions_aug = []
    kpt = KeypointsOnImage([
        Keypoint(x=position[0],y=position[1])
    ], shape=image.shape)
    
    for _ in range(aug_multiplier):
        img_aug, kpt_aug = AUG_SEQ(image=image, keypoints=kpt)
        pos_aug = np.array([kpt_aug[0].x, kpt_aug[0].y], dtype=int)
        images_aug.append(img_aug)
        positions_aug.append(pos_aug)
    return images_aug, positions_aug
        


def augment_data(image, grasp, label, aug_rot_itvl=4, aug_multiplier=4):
    images_aug = []
    grasps_aug = []
    labels_aug = []

    _rots = []
    for i in range(len(label)):
        _rots.append(i*(360/len(label)))
    # # generate image array
    kps = KeypointsOnImage([
        Keypoint(x=grasp[0][0],y=grasp[0][1]),
        Keypoint(x=grasp[1][0],y=grasp[1][1])
    ], shape=image.shape)
    
    for i in range(aug_rot_itvl): 
        rot_degree = i*(360/aug_rot_itvl)
        seq = iaa.Sequential([
                iaa.Affine(
                    scale=(0.9,1.1),
                    shear=(-10,10),
                    rotate=rot_degree
                ),
                iaa.GammaContrast((0.5, 2.0)),
                iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=0.5, sigma=0.5))
                ])
        # images = np.array([drawn for _ in range(4)], dtype=np.uint8)
        # images_aug = seq(images=images)
        for _ in range(aug_multiplier):
            img_aug, kps_aug = seq(image=image, keypoints=kps)
            grasp_aug = []
            for k in range(len(kps.keypoints)):
                grasp_aug.append([int(kps_aug[k].x), int(kps_aug[k].y)])
            image_after = kps_aug.draw_on_image(img_aug, size=7)
            # ia.imshow(image_after)
            search_degree = (360-rot_degree) % 360 
            lbl_aug = label[_rots.index(search_degree):] + label[:_rots.index(search_degree)]
            images_aug.append(img_aug)
            # images_aug.append(image_after)
            grasps_aug.append(grasp_aug)
            labels_aug.append(lbl_aug)
            # cv2.circle(img_aug, grasp_aug[0], 7, (0,255,0), -1)
            # cv2.imshow("", img_aug)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
    return images_aug, grasps_aug, labels_aug

if __name__ == "__main__":
    # ---------------- generate original data ------------------
    # src_dir = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_final_fine"
    src_dir = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\data_final_relabel"
    src_dir = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\data_final_real"
    # aug_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorAugment"
    aug_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorVal"

    # # ------------- various shapes -------------------- 
    # for _s in os.listdir(src_dir):
    #     if _s[0] == '_': continue
    #     if _s.upper() != 'U': continue
    #     _src_dir = os.path.join(src_dir, _s)
    #     print('--------', _src_dir, " => ", aug_dir, '--------')
    #     gen_sepdata_from_pe(_src_dir, aug_dir, itvl=8)
    
    # # ------------ single folder ---------------------
    # print('--------', src_dir, " => ", aug_dir, '--------')
    # gen_sepdata_from_pe(src_dir, aug_dir, itvl=8)
    

    # ------------- generate data from oc + sln.json --------------
    # src_dir = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes_relabel"
    # aug_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEight"
    # for _s in os.listdir(src_dir):
    #     _src_dir = os.path.join(src_dir, _s)
    #     print("--------", _src_dir, " => ", aug_dir, "--------")
    #     gen_sepdata_from_oc(_src_dir, aug_dir, itvl=8)


    # ------------- generate data for picknet --------------
    # src_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\TangleData"
    # aug_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\PickDataNew"
    # for _s in os.listdir(src_dir):
    #     if "_C" in _s or "_E" in _s or "NEW" in _s:
    #         continue
    #     _src_dir = os.path.join(src_dir, _s)
    #     print('----------------------------------------')
    #     print('|  ', _src_dir, '\n|=>', aug_dir)
    #     gen_pickdata(_src_dir, aug_dir)
    #     print('----------------------------------------')
    
    # ------------- Simplify pick data: remove json and rename--------------
    # # from exp
    # src_dir = "C:\\Users\\xinyi\\Desktop\\exp"
    # dest_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\PickDataNew"
    # gen_simple_pickdata(src_dir, dest_dir, type="sep", aug_multiplier=1) 

    # # from TangleData
    # src_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\TangleData"
    # dest_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\PickDataNew"
    # for _s in os.listdir(src_dir):
    #     if "_C" in _s or "_E" in _s or "NEW" in _s or "NEW" in _s:
    #         continue
    #     _src_dir = os.path.join(src_dir, _s)
    #     print('----------------------------------------')
    #     print('|  ', _src_dir, '\n|=>', dest_dir)
    #     gen_simple_pickdata(_src_dir, dest_dir)
    #     print('----------------------------------------')


    # ------------- generate data for simplified sepnet --------------
    src_dir = "C:\\Users\\xinyi\\Desktop\\exp"
    dest_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataNewAug"
    gen_simple_sepdata(src_dir, dest_dir, aug_multimplier=4) 
    