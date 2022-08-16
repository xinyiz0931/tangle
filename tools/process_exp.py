import numpy as np
import cv2
import os
import json
import glob
d = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEight\\positions.npy"
img_path = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEight\\images\\000000.png"
_img = cv2.imread(img_path)
img_w, img_h = 640, 480
_h, _w, _ = _img.shape
_p = np.load(d)[0]
cv2.circle(_img, _p[0], 7, (0,255,0), -1)
cv2.circle(_img, _p[1], 7, (0,255,0), 2)
cv2.imshow("", _img)
cv2.waitKey()
cv2.destroyAllWindows()
print("(B): ", _p.flatten())
_p[:,0] = _p[:,0] * img_w / _w
_p[:,1] = _p[:,1] * img_h / _h
p = _p.astype(int)

img = cv2.resize(_img, (img_w, img_h))
print("(A): ", p.flatten())
cv2.circle(img, p[0], 7, (0,0,255), -1)
cv2.circle(img, p[1], 7, (0,0,255), 2)

cv2.imshow("", img)
cv2.waitKey()
cv2.destroyAllWindows()