import json
import os
import cv2
import glob
import math
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import parameters as iap

seq = iaa.BlendAlphaSimplexNoise(
    iaa.InvertMaskGen(0.2, iaa.VerticalLinearGradientMaskGen()),
    iaa.Clouds()
)
seq = iaa.BlendAlphaFrequencyNoise(
    exponent=math.e,
    foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
    size_px_max=32,
    upscale_method="linear",
    iterations=1,
    sigmoid=False
)
ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
src = ia.quokka(size=(64,64))
src = cv2.imread("C:\\Users\\xinyi\\Documents\\XYBin_Collected\\tangle_scenes_relabel\\U\\23\\depth.png")

images = np.array(
    [src for _ in range(16)],
    dtype=np.uint8
)
images_aug = seq(images=images)
ia.show_grid(images_aug, rows=2)