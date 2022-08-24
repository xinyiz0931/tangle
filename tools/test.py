import torch
from torchvision.transforms.functional import rotate
from tangle.utils import *
from bpbot.utils import rotate_img
itvl = 8

# print("Manually: ", out_m.shape)

img_path = "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes\\SC\\35\\depth.png" 
img = cv2.imread(img_path, 0)

img = draw_vector(img, (250,250), (0,1))
print(img.shape)
img_r = rotate_img(img, 90)
[h1, h2] = gauss_2d_batch(500,500,9, [[250,250], [300,300]])
h1 = visualize_tensor(h1)
h2 = visualize_tensor(h2)
print(h1.shape, h2.shape)

cv2.imshow("", cv2.hconcat([h1, h2]))
cv2.waitKey()
cv2.destroyAllWindows()