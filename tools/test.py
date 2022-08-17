import torch
import cv2
import numpy as np
src_t = torch.rand((7, 50,500))


from tangle.utils import visualize_tensor
ret = visualize_tensor(src_t, cmap=False)
print(src_t.shape, "=>", ret.shape)
# cv2.imshow("w", ret)
# cv2.waitKey()
# cv2.destroyAllWindows()