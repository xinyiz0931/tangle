import os
import numpy as np
import cv2
import glob
from tangle.utils import *

def p2v(clicked_p, p_pull, itvl=8, l=50):
    stop_ps = []
    degrees = []
    for r in range(itvl):
        degree = r * (360/itvl)
        degrees.append(degree)
        v = angle2vector(degree)
        stop_ps.append([int(p_pull[0] + v[0]*l), int(p_pull[1] + v[1]*l)])
    min_dist = 9999
    min_r = None
    for r, _q in zip(degrees, stop_ps): 
        dist = np.linalg.norm(np.array(_q) - np.array(clicked_p))
        if dist <= min_dist: 
            min_r = r
            min_dist = dist
    return min_r
    # print("Selected vector p: ", min_r)
    
def calc_v(clicked_p, p_pull, itvl=8, l=50):
    stop_ps = []
    degrees = []
    for r in range(itvl):
        degree = r * (360/itvl)
        degrees.append(degree)
        v = angle2vector(degree)
        stop_ps.append([int(p_pull[0] + v[0]*l), int(p_pull[1] + v[1]*l)])
    for _p in clicked_p:
        min_dist = 9999
        min_r = None
        for r, _q in zip(degrees, stop_ps): 
            dist = np.linalg.norm(np.array(_q) - np.array(_p))
            if dist <= min_dist: 
                min_r = r
                min_dist = dist
        print("Selected vector p: ", min_r)


def draw(src, grasps, itvl=8, l=50):
    img = src.copy()
    # grasps: p_pull, p_hold
    p_hold = grasps[0]
    p_pull = grasps[1]
    cv2.circle(img, p_hold, 5, (0,255,255), 2)
    cv2.circle(img, p_pull, 5, (0,255,0), -1)
    cv2.circle(img, p_pull, l, (167,185,175), 2)
    for r in range(itvl):
        degree = r* (360/itvl)
        v = angle2vector(degree) 
        p_stop = [int(p_pull[0] + v[0]*l), int(p_pull[1] + v[1]*l)]
        
        cv2.line(img, p_pull, p_stop, (167,185,175), 1)
        cv2.circle(img, p_stop, 9, (167,185,175),-1)
    return img
    cv2.imshow("windows", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


root_dir = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_exp_for_rsj2022\\TaskEmpty"
_search = os.path.join(root_dir, "**\\depth_mid*.png")
i = 0
for d in glob.glob(_search, recursive=True):
    print(d)
    pass_flag = False
    save_dir = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes_relabel\\EXP\\%.4d" % i
    img_s_path = os.path.join(save_dir, "depth.png")
    json_s_path = os.path.join(save_dir, "sln.json")
    draw_s_path = os.path.join(save_dir, "ret.png")
    
    img = cv2.imread(d)
    drawn = img.copy()
    grasps = []
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(drawn, (x,y), 5, (0,255,0), -1)
            print(f"------ click hold-and-pull: ({x}, {y})")
            grasps.append([x,y])
    cv2.namedWindow("w")
    cv2.setMouseCallback("w", on_click)
    while (len(grasps) < 2):
        cv2.imshow("w", drawn)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('r'):
            grasps = []
            drawn = img.copy()
        elif k == ord('p'):
            pass_flag = True
            break
    if not pass_flag:
        ret = draw(img, grasps)
        drawn = ret.copy()
        sln_angles = []
        def on_click2(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(drawn, (x,y), 5, (0,255,0), -1)
                angle = p2v([x,y], grasps[1])
                print(f"------ click pull vector: ({angle})")
                sln_angles.append(angle)
        cv2.namedWindow("v")
        cv2.setMouseCallback("v", on_click2)
        while(True):
            k = cv2.waitKey(1) & 0xFF
            cv2.imshow("v", drawn)
            if k == ord('r'):
                sln_angles = []
                drawn = ret.copy()
            elif k == 13: # enter
                break
        final_ret = img.copy()
        cv2.circle(final_ret, grasps[0], 5 ,(0,255,255), 2)
        cv2.circle(final_ret, grasps[1], 7, (0,255,0), -1)
        for r in sln_angles:
            final_ret = draw_vector(final_ret, grasps[1], angle2vector(r))
        print(f"[{i:>4}] Hold and pull point: {grasps}, rotations: {sln_angles}")
        j_file = {"hold": [], "pull": [], "angle": []}
        j_file["hold"] = grasps[0]
        j_file["pull"] = grasps[1]
        j_file["angle"] = sln_angles
        if not os.path.exists(save_dir): 
            os.mkdir(save_dir)
            cv2.imwrite(img_s_path, img)
            cv2.imwrite(draw_s_path, final_ret)
            with open(json_s_path, 'w') as f:
                json.dump(j_file, f, indent=4)
            i += 1
        cv2.imshow("r", final_ret)
        cv2.waitKey()
        cv2.destroyAllWindows()

print(f"Total {i} samples transferred! ")

# root_dir = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes_relabel\\*"
# i=0
# for d in glob.glob(root_dir):
#     shape = d.split('\\')[-1]
#     if shape != 'SC': continue
#     for f in glob.glob(os.path.join(d, '*')):
#         f_idx = f.split('\\')[-1]
#         img_path = os.path.join(f, "depth.png")
#         ret_path = os.path.join(f, "ret.png")
        
#         j_path = os.path.join(f, "sln.json")

#         if os.path.exists(j_path):
#             continue
#         # if i > 5: 
#         #   break
#         pass_flag = False
#         j_file = {} 
#         img = cv2.imread(img_path)
#         drawn = img.copy()
#         grasps = []
#         def on_click(event, x, y, flags, param):
#             if event == cv2.EVENT_LBUTTONDOWN:
#                 cv2.circle(drawn, (x,y), 5, (0,255,0), -1)
#                 print(f"------ click hold-and-pull: ({x}, {y})")
#                 grasps.append([x,y])
#         cv2.namedWindow("hold first, then pull")
#         cv2.setMouseCallback("hold first, then pull", on_click)
#         while(len(grasps) < 2):
#             cv2.imshow("hold first, then pull", drawn)
#             k = cv2.waitKey(20) & 0xFF
#             if k == ord('r'):
#                 grasps = []
#                 drawn = img.copy()
#             elif k == ord('p'):
#                 pass_flag = True
#                 break
#         if not pass_flag: 
#             ret = draw(img, grasps) 
#             drawn = ret.copy()
#             sln_angles = []
#             def on_click2(event, x, y, flags, param):
#                 if event == cv2.EVENT_LBUTTONDOWN:
#                     cv2.circle(drawn, (x,y), 5,(0,255,0),-1)
#                     angle = p2v([x,y], grasps[1])
#                     print(f"------ click pull vector: ({angle})")
#                     sln_angles.append(angle) 
#             cv2.namedWindow("pull vector")
#             cv2.setMouseCallback("pull vector", on_click2)
#             while(True): 
#                 k = cv2.waitKey(1) & 0xFF
#                 cv2.imshow("pull vector", drawn)
#                 if k == ord('r'): 
#                     sln_angles = []
#                     drawn = ret.copy()
#                 elif k == 13: # enter
#                     break
            
#             # calc_v(vectors, grasps[1])
#             final_ret = img.copy() 
#             cv2.circle(final_ret, grasps[0], 5, (0,255,255), 2)
#             cv2.circle(final_ret, grasps[1], 7, (0,255,0), -1)
#             for r in sln_angles: final_ret = draw_vector(final_ret, grasps[1], angle2vector(r))
#             cv2.imshow("result", final_ret)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
#             cv2.imwrite(ret_path, final_ret)
#             print(f"[{f_idx:>4}] Hold and pull point: {grasps}, rotations: {sln_angles}")
#             j_file = {"hold": [], "pull": [], "angle": []}
#             j_file["hold"] = grasps[0]
#             j_file["pull"] = grasps[1]
#             j_file["angle"] = sln_angles
#         with open(j_path, 'w') as f:
#             json.dump(j_file, f, indent=4)
#         i += 1

#     break
        

