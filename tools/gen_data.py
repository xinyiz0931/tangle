
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
from tangle.utils import *
from bpbot.utils import *

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
    for i, pi in enumerate(poses_sln):
        for j, pj in enumerate(poses_src):
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
        num = 0
        grasps, directions, solutions = [], [], []
        for p in pos_list:
            if not p in grasps:
                grasps.append(p)
        
        for g in grasps:
            _rot = []
            _sln = []
            for index, p in enumerate(pos_list):
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
    num = 0
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

            for i, g in enumerate(grasps):
                drawn = img.copy()
                
                for j in range(itvl):
                    r = j*(360/itvl)
                    if r in directions[i]:
                        draw_vector(drawn, (g[0], g[1]), direction2vector(r), 40, arrow_thinkness=2,color=(0,255,0))
                    else:
                        draw_vector(drawn, (g[0], g[1]), direction2vector(r), 40, arrow_thinkness=1,color=(0,0,255))
                drawn = cv2.circle(drawn, (g[0],g[1]),7,(0,255,0),-1)
                # drawn = cv2.circle(drawn, (g[2],g[3]),7,(0,255,0),-1)
                save_path = os.path.join(dest_dir, "%06d.png"%num)
                cv2.imwrite(save_path, drawn)
                num += 1
                #cv2.imshow(s, drawn)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

# def gen_sepdata_from_oc(source_dir, dest_dir, itvl=16):
#     """
#     Generate dastaset from only calculated data, using directory/info.json
#     ├ dest_dir
#     ├── images
#     │   ├── 000000.png
#     │   └── ...
#     ├── positions.npz - np.array([[pull_x, pull_y, hold_x, hold_y], [...], ...]), shape = (16 x N)
#     └── direction.npz - np.array([[1,0,0,0,...], [...], ...]), shape=(16 x N)
#     """
#     images_dir = os.path.join(dest_dir, 'images')
#     positions_path = os.path.join(dest_dir, 'positions.npy')
#     labels_path = os.path.join(dest_dir, 'labels.npy')
#     direction_path = os.path.join(dest_dir, 'direction.npy')
    
#     if not os.path.exists(dest_dir): os.mkdir(dest_dir)
#     if not os.path.exists(images_dir): os.mkdir(images_dir)
    
#     num = 0
#     num_exist = len(os.listdir(images_dir))
    
#     if os.path.exists(positions_path):
#         positions_list = np.load(positions_path).tolist()
#     else: 
#         positions_list = []
#     if os.path.exists(labels_path):
#         labels_list = np.load(labels_path).tolist()
#     else:
#         labels_list = []

#     direction_list = []
#     for i in range(itvl): 
#         direction_list.append(angle2vector(i*360/itvl))

#     print(f"Total {len(os.listdir(source_dir))} samples! ")
#     for data in os.listdir(source_dir):
#         d= os.path.join(source_dir, data)
#         j_path = os.path.join(d, 'sln.json')
#         if not os.path.exists(j_path): continue
#         f = open(j_path, 'r+')
#         j = json.loads(f.read())
#         img = cv2.imread(os.path.join(d, 'depth.png'))
#         l = [0]*itvl
#         # if "pullhold" in j:
#         #     for angle in j['angle']:
#         #         angle += 180
#         #         angle %= 360
#         #         d_index = int(angle/(360/itvl))
#         #         l[d_index] = 1
#         #     g = np.reshape(j['pullhold'], (2,2))
            
#         if "pull" in j and "hold" in j:
#             for angle in j["angle"]:
#                 d_idx = int(angle/(360/itvl))
#                 l[d_idx] = 1
#                 g = np.array([j["pull"], j["hold"]])
#             i_list, g_list, l_list = augment_data(image=img, grasp=g, label=l, aug_rot_itvl=4, aug_multiplier=3)
#             for i_, g_, l_ in zip(i_list, g_list, l_list):
#                 positions_list.append(g_)
#                 labels_list.append(l_)
#                 new_img_path = os.path.join(images_dir, '%06d.png'%(num+num_exist))
#                 cv2.imwrite(new_img_path, i_)
#                 num += 1
#         # if num % 100 == 0: 
#         #     print(f"Transfer {num} samples! ") 
#     np.save(positions_path, positions_list)
#     np.save(labels_path, labels_list)
#     np.save(direction_path, direction_list)

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
    
    num = 0
    num_exist = len(os.listdir(images_dir))
    
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
                for i, g in enumerate(grasps):
                    g = np.reshape(g, (2,2))
                    for j in range(itvl):
                        r = j*(360/itvl)
                        if r in degrees[i]:
                            l[j] = 1
        
        elif "pull" in json_file and "hold" in json_file:
            degrees = json_file["angle"]
            g = np.array([j["pull"], j["hold"]])
            for j in range(itvl):
                r = j*(360/itvl)
                if r in degrees: 
                    l[j] = 1

        i_list, g_list, l_list = augment_data(image=img, grasp=g, label=l, aug_rot_itvl=4, aug_multiplier=3) 
        for i_, g_, l_ in zip(i_list, g_list, l_list):
            new_img_path = os.path.join(images_dir, "%06d.png" % (num+num_exist))
            positions_list.append(g_)
            labels_list.append(l_)
            cv2.imwrite(new_img_path, i_)
            num += 1

            if num % 100 == 0: print(f"Transferred {num} samples! ")
    np.save(positions_path,positions_list)
    np.save(labels_path, labels_list)
    np.save(direction_path, direction_list) 
        
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
                    shear=(-5,5),
                    rotate=rot_degree
                ),
                iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=0.5, sigma=0.5))
                ])
        # images = np.array([drawn for _ in range(4)], dtype=np.uint8)
        # images_aug = seq(images=images)
        for _ in range(aug_multiplier):
            img_aug, kps_aug = seq(image=image, keypoints=kps)
            grasp_aug = []
            for k in range(len(kps.keypoints)):
                grasp_aug.append([int(kps_aug[k].x), int(kps_aug[k].y)])
            #  image_after = kps_aug.draw_on_image(img_aug, size=7)
            # ia.imshow(image_after)
            search_degree = (360-rot_degree) % 360 
            lbl_aug = label[_rots.index(search_degree):] + label[:_rots.index(search_degree)]
            images_aug.append(img_aug)
            grasps_aug.append(grasp_aug)
            labels_aug.append(lbl_aug)
            # cv2.circle(img_aug, grasp_aug[0], 7, (0,255,0), -1)
            # cv2.imshow("", img_aug)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
    return images_aug, grasps_aug, labels_aug

if __name__ == "__main__":
    # ---------------- generate original data ------------------
    src_dir = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_final_fine"
    # src_dir = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_only_calc"
    aug_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorAugment"
    aug_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEight"
    for _s in os.listdir(src_dir):
        if _s[0] == '_': continue
        if _s.upper() != 'SR': continue
        _src_dir = os.path.join(src_dir, _s)
        print('--------', _src_dir, " => ", aug_dir, '--------')
        # gen_sepdata_from_pe(_src_dir, aug_dir)
        gen_sepdata_from_pe(_src_dir, aug_dir, itvl=8)

    # ------------- generate data from oc + sln.json --------------
    # src_dir = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes_relabel"
    # aug_dir = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEight"
    # for _s in os.listdir(src_dir):
    #     _src_dir = os.path.join(src_dir, _s)
    #     print("--------", _src_dir, " => ", aug_dir, "--------")
    #     gen_sepdata_from_oc(_src_dir, aug_dir, itvl=8)


    