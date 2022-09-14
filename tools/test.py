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


    


# print("Manually: ", out_m.shape)
p = np.array([[6,5], [7,4]])
# p = np.array([[6,5]])

np_gauss_2d_batch(12,10,3,p)



g = gauss_2d_batch(50, 50, 8, p, use_torch=False)
print(g.shape)
# vis = visualize_tensor(g)
# cv2.imshow("", vis)
# cv2.waitKey()
# cv2.destroyAllWindows()
# print(vis.shape)
# img_path = "C:\\Users\\xinyi\\Pictures\\t.png"
# img_path = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\data_final_real\\0001\\depth.png"
# img = cv2.imread(img_path)
# g = np.array([[242,340], [140,319]])
# l = [0,0,0,0,0,0,0,1]


# img = draw_vectors_bundle(img, g[0], scores=l)
# cv2.circle(img, g[0], 7, (0,255,0), -1)
# cv2.circle(img, g[1], 7, (0,255,255), 2)
# cv2.imshow("", img)
# cv2.waitKey()
# cv2.destroyAllWindows()